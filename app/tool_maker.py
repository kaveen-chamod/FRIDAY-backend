from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import tempfile
import threading
import time

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from app.access_control import assert_safe_write_target
from core.sandbox import scan_for_danger


PROJECT_ROOT = Path(__file__).resolve().parents[1]

STAGING_DIR = Path(
    os.environ.get(
        "FRIDAY_STAGING_DIR",
        PROJECT_ROOT / "staging_tools",
    )
).expanduser().resolve()

MANIFEST_FILE = PROJECT_ROOT / "memory" / "tool_manifest.json"

MAX_CAPABILITY_LENGTH = 500
MAX_SOURCE_LENGTH = 100_000
SANDBOX_TIMEOUT_SECONDS = 10
MAX_REPAIR_ATTEMPTS = 3

_MANIFEST_LOCK = threading.RLock()


_FORBIDDEN_IMPORTS = {
    "builtins",
    "ctypes",
    "importlib",
    "multiprocessing",
    "os",
    "pathlib",
    "pickle",
    "pty",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "sys",
    "tempfile",
    "urllib",
}

# psutil ආරක්ෂිතව එක් කර ඇත (System health tools හැදීමට ඉඩ දීම සඳහා)
_ALLOWED_IMPORTS = {
    "app.modules.base",
    "collections",
    "datetime",
    "decimal",
    "enum",
    "json",
    "math",
    "re",
    "statistics",
    "typing",
    "psutil",
}

_FORBIDDEN_CALLS = {
    "__import__",
    "compile",
    "eval",
    "exec",
    "open",
    "getattr",
    "setattr",
    "delattr",
    "globals",
    "locals",
    "vars",
    "breakpoint",
    "input",
}

_FORBIDDEN_ATTRIBUTES = {
    "__builtins__",
    "__class__",
    "__code__",
    "__dict__",
    "__globals__",
    "__mro__",
    "__subclasses__",
    "connect",
    "kill",
    "link",
    "open",
    "popen",
    "remove",
    "rename",
    "rmdir",
    "system",
    "unlink",
    "write_bytes",
    "write_text",
}


class ToolMaker:
    """
    F.R.I.D.A.Y. AI Autonomous Tool Maker with Approval, Staging, & Deployment Lifecycle.
    """

    def __init__(
        self,
        generate_text: Callable[[str], str],
        notify_approval: Callable[[dict], Any] | None = None,
        verify_approval: Callable[[str, str, str], bool] | None = None,
        deploy_staged_tool: Callable[[Path, dict], Any] | None = None,
    ) -> None:

        if not callable(generate_text):
            raise TypeError("generate_text must be callable.")

        self._generate_text = generate_text
        self._notify_approval = notify_approval
        self._verify_approval = verify_approval
        self._deploy_staged_tool = deploy_staged_tool

    # ---------------------------------------------------------
    # 1. MAIN TOOL CREATION WORKFLOW
    # ---------------------------------------------------------

    def stage(
        self,
        capability: str,
        action_registry=None,
        plugin_registry=None,
    ) -> Path:
        capability = capability.strip()

        if not capability or len(capability) > MAX_CAPABILITY_LENGTH:
            raise ValueError(
                f"Describe one capability in 1 to {MAX_CAPABILITY_LENGTH} characters."
            )

        specification = self.analyze(
            capability,
            action_registry=action_registry,
            plugin_registry=plugin_registry,
        )

        spec_prompt = self._build_generation_prompt(specification)

        source = ""
        diagnostic = ""
        verification: dict[str, Any] = {}

        for attempt in range(MAX_REPAIR_ATTEMPTS + 1):
            if attempt == 0:
                prompt = spec_prompt
            else:
                prompt = (
                    f"{spec_prompt}\n\n"
                    "The previous candidate failed validation.\n"
                    f"Reason: {diagnostic}\n\n"
                    "Previous candidate:\n"
                    f"{source}\n\n"
                    "Return a corrected complete Python module only."
                )

            generated = self._generate_text(prompt)

            if not isinstance(generated, str):
                raise ValueError("The AI code generator returned an invalid response.")

            source = self._clean_source(generated)

            if not source or len(source) > MAX_SOURCE_LENGTH:
                diagnostic = "Generated source is empty or too large."
                continue

            try:
                self._validate(source)
                metadata = self._tool_metadata(source)
                self._validate_metadata_against_spec(metadata, specification)
                verification = self._sandbox_test(source, specification)
                break

            except (ValueError, RuntimeError) as exc:
                diagnostic = str(exc)

        else:
            raise ValueError(
                f"Tool proposal failed validation after {MAX_REPAIR_ATTEMPTS} repair attempts: {diagnostic}"
            )

        name = specification["name"]

        assert_safe_write_target(STAGING_DIR)
        STAGING_DIR.mkdir(parents=True, exist_ok=True)

        destination = STAGING_DIR / f"{name}.py"
        assert_safe_write_target(destination)

        if destination.exists():
            raise FileExistsError(f"A staged proposal named '{name}' already exists.")

        try:
            with destination.open("x", encoding="utf-8") as stream:
                stream.write(source + "\n")
                stream.flush()
                os.fsync(stream.fileno())

            if os.name != "nt":
                destination.chmod(0o600)

            file_hash = self._sha256(destination)

            entry = self._record_staged_tool(
                name=name,
                capability=capability,
                specification=specification,
                destination=destination,
                verification=verification,
                file_hash=file_hash,
            )

        except Exception:
            assert_safe_write_target(destination)
            destination.unlink(missing_ok=True)
            raise

        if self._notify_approval is not None:
            try:
                self._notify_approval(
                    {
                        "event": "tool_pending_approval",
                        "name": name,
                        "version": entry["version"],
                        "description": specification["description"],
                        "path": str(destination),
                        "status": "pending_approval",
                        "message": "A new AI-generated tool is ready for review.",
                    }
                )
            except Exception:
                pass

        return destination

    # ---------------------------------------------------------
    # 2. REQUIREMENT ANALYSIS
    # ---------------------------------------------------------

    def analyze(
        self,
        capability: str,
        action_registry=None,
        plugin_registry=None,
    ) -> dict:
        existing: list[dict] = []

        if action_registry is not None:
            existing.extend(action_registry.get_tool_declarations())

        if plugin_registry is not None:
            existing.extend(plugin_registry.get_tool_declarations())

        normalized_existing = []
        for item in existing:
            if not isinstance(item, dict):
                continue
            normalized_existing.append(
                {
                    "name": str(item.get("name", "")),
                    "description": str(item.get("description", "")),
                }
            )

        existing_names = sorted(
            {tool["name"] for tool in normalized_existing if tool["name"]}
        )

        inventory = {
            "request": capability,
            "language_note": "Understand English, Sinhala and Singlish.",
            "candidate_dependencies": ["psutil"],
            "existing_tool_names": existing_names,
        }

        analysis_prompt = (
            "You are F.R.I.D.A.Y.'s autonomous tool requirement analyzer.\n"
            "Analyze the requested capability before writing code.\n"
            "Return ONLY one valid JSON object.\n\n"
            "Required keys:\n"
            "- name\n"
            "- description\n"
            "- parameters\n"
            "- expected_output\n"
            "- candidate_dependencies\n"
            "- duplicate_of\n\n"
            "Rules:\n"
            "1. Tool name must be a valid unique Python identifier in snake_case (e.g. windows_health_monitor).\n"
            "2. Parameters must be a JSON object schema.\n"
            "3. Dependencies must use approved standard-library modules or 'psutil'.\n"
            "4. Do not generate code.\n"
            "5. If it directly duplicates an existing tool, set duplicate_of to that tool's name; otherwise set duplicate_of to null.\n\n"
            f"Inventory:\n{json.dumps(inventory, ensure_ascii=False)}"
        )

        raw = self._generate_text(analysis_prompt)
        if not isinstance(raw, str):
            raise ValueError("Requirement analyzer returned an invalid response.")

        raw = self._clean_source(raw)

        try:
            specification = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("Requirement analysis did not return valid JSON.") from exc

        if not isinstance(specification, dict):
            raise ValueError("Requirement analysis must return one JSON object.")

        name = specification.get("name")
        description = specification.get("description")
        parameters = specification.get("parameters")
        expected_output = specification.get("expected_output")
        dependencies = specification.get("candidate_dependencies") or []
        duplicate = specification.get("duplicate_of")

        if not isinstance(name, str) or not re.fullmatch(r"[a-zA-Z_][a-zA-Z0-9_]{0,63}", name):
            raise ValueError("Requirement analysis returned an invalid tool name.")

        if name in existing_names:
            name = f"{name}_custom"
            specification["name"] = name

        if not isinstance(description, str) or not description.strip():
            raise ValueError("Requirement analysis returned an invalid description.")

        self._validate_parameter_schema(parameters)

        if not isinstance(expected_output, str) or not expected_output.strip():
            specification["expected_output"] = "Structured health diagnostic report string"

        specification.update(
            {
                "request": capability,
                "existing_tool_names": existing_names,
            }
        )

        return specification

    # ---------------------------------------------------------
    # 3. GENERATION PROMPT
    # ---------------------------------------------------------

    @staticmethod
    def _build_generation_prompt(specification: dict) -> str:
        return (
            "Create one small, robust Python tool for F.R.I.D.A.Y.\n\n"
            "Use this exact import:\n"
            "from app.modules.base import BaseModule\n\n"
            "Requirements:\n"
            "1. Create exactly one BaseModule subclass.\n"
            "2. Declare a literal class-level TOOL dictionary with name, description, parameters.\n"
            "3. Implement run(self, parameters: dict) -> str.\n"
            "4. Implement execute(self, **kwargs) -> str as an adapter to run().\n"
            "5. Validate all input parameters.\n"
            "6. Return a useful, non-empty formatted string.\n"
            "7. Return Python source only. No markdown formatting.\n\n"
            f"Specification:\n{json.dumps(specification, ensure_ascii=False, indent=2)}"
        )

    # ---------------------------------------------------------
    # 4. METADATA VALIDATION
    # ---------------------------------------------------------

    @staticmethod
    def _validate_metadata_against_spec(metadata: dict, specification: dict) -> None:
        if metadata["name"] != specification["name"]:
            raise ValueError("Generated tool name does not match specification.")
        if not metadata["description"].strip():
            raise ValueError("Generated tool description is empty.")

    @staticmethod
    def _validate_parameter_schema(parameters: Any) -> None:
        if not isinstance(parameters, dict):
            raise ValueError("Parameter schema must be an object.")
        if parameters.get("type") != "OBJECT":
            raise ValueError("Parameter schema type must be OBJECT.")

    # ---------------------------------------------------------
    # 5. SOURCE CLEANING
    # ---------------------------------------------------------

    @staticmethod
    def _clean_source(source: str) -> str:
        cleaned = source.strip()
        cleaned = re.sub(r"^```(?:python|json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        return cleaned.strip()

    # ---------------------------------------------------------
    # 6. STATIC CODE VALIDATION
    # ---------------------------------------------------------

    @staticmethod
    def _validate(source: str) -> None:
        if not source or len(source) > MAX_SOURCE_LENGTH:
            raise ValueError("Source is empty or exceeds the allowed size.")

        try:
            tree = ast.parse(source, filename="<staged-tool>")
            compile(tree, "<staged-tool>", "exec")
        except SyntaxError as exc:
            raise ValueError(f"Generated proposal has invalid syntax: {exc}") from exc

        base_import = any(
            isinstance(node, ast.ImportFrom)
            and node.module == "app.modules.base"
            and any(alias.name == "BaseModule" for alias in node.names)
            for node in tree.body
        )

        base_classes = [
            node for node in tree.body
            if isinstance(node, ast.ClassDef)
            and any(isinstance(base, ast.Name) and base.id == "BaseModule" for base in node.bases)
        ]

        if not base_import or len(base_classes) != 1:
            raise ValueError("Proposal must import and define exactly one BaseModule subclass.")

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root_module = alias.name.split(".", 1)[0]
                    if root_module in _FORBIDDEN_IMPORTS:
                        raise ValueError(f"Prohibited import: {root_module}")
                    if root_module not in _ALLOWED_IMPORTS:
                        raise ValueError(f"Unsupported import: {root_module}")

            if isinstance(node, ast.ImportFrom):
                module_name = node.module or ""
                root_module = module_name.split(".", 1)[0]
                if root_module in _FORBIDDEN_IMPORTS:
                    raise ValueError(f"Prohibited import: {root_module}")
                if root_module not in _ALLOWED_IMPORTS:
                    raise ValueError(f"Unsupported import: {module_name}")

            if isinstance(node, ast.Attribute):
                if node.attr in _FORBIDDEN_ATTRIBUTES:
                    raise ValueError(f"Prohibited attribute access: {node.attr}")
                if node.attr.startswith("__"):
                    raise ValueError("Dunder attribute access is not permitted.")

            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id in _FORBIDDEN_CALLS:
                    raise ValueError(f"Prohibited function call: {node.func.id}")

        candidate_class = base_classes[0]
        methods = {
            node.name for node in candidate_class.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

        if "run" not in methods or "execute" not in methods:
            raise ValueError("The BaseModule subclass must implement run() and execute().")

        hazards = scan_for_danger(source)
        if hazards:
            raise ValueError("Static safety scan detected: " + ", ".join(map(str, hazards)))

    # ---------------------------------------------------------
    # 7. TOOL METADATA
    # ---------------------------------------------------------

    @staticmethod
    def _tool_metadata(source: str) -> dict:
        module = ast.parse(source)
        for node in module.body:
            if not isinstance(node, ast.ClassDef):
                continue
            for item in node.body:
                if not isinstance(item, ast.Assign):
                    continue
                if not any(isinstance(target, ast.Name) and target.id == "TOOL" for target in item.targets):
                    continue
                try:
                    metadata = ast.literal_eval(item.value)
                except (ValueError, TypeError) as exc:
                    raise ValueError("TOOL must be a literal dictionary.") from exc

                if isinstance(metadata, dict) and metadata.get("name"):
                    return metadata

        raise ValueError("Proposal must declare a literal TOOL dictionary.")

    # ---------------------------------------------------------
    # 8. SANDBOX TESTING
    # ---------------------------------------------------------

    @staticmethod
    def _sandbox_test(source: str, specification: dict) -> dict:
        from core.sandbox import run_sandboxed

        parameters = _mock_parameters(specification["parameters"])

        runner = (
            "import json, sys, types, unittest\n"
            "app = types.ModuleType('app'); app.__path__ = []\n"
            "modules = types.ModuleType('app.modules'); modules.__path__ = []\n"
            "base = types.ModuleType('app.modules.base')\n"
            "class BaseModule: pass\n"
            "base.BaseModule = BaseModule\n"
            "sys.modules.update({'app': app, 'app.modules': modules, 'app.modules.base': base})\n"
            f"candidate_source = {source!r}\n"
            f"mock_parameters = json.loads({json.dumps(json.dumps(parameters))})\n"
            "namespace = {'__name__': 'candidate_tool'}\n"
            "exec(compile(candidate_source, '<candidate-tool>', 'exec'), namespace)\n"
            "candidates = [v for v in namespace.values() if isinstance(v, type) and issubclass(v, BaseModule) and v is not BaseModule]\n"
            "if len(candidates) != 1: raise RuntimeError('Expected one tool class.')\n"
            "tool = candidates[0]()\n"
            "class CandidateToolTests(unittest.TestCase):\n"
            "    def test_run(self):\n"
            "        result = tool.run(mock_parameters)\n"
            "        self.assertIsInstance(result, str)\n"
            "suite = unittest.defaultTestLoader.loadTestsFromTestCase(CandidateToolTests)\n"
            "result = unittest.TextTestRunner(verbosity=0).run(suite)\n"
            "if not result.wasSuccessful(): raise RuntimeError('Candidate tests failed.')\n"
            "print('TOOL_TESTS_PASSED')\n"
        )

        with tempfile.TemporaryDirectory(prefix="friday-tool-test-") as directory:
            runner_path = Path(directory) / "candidate_test_runner.py"
            assert_safe_write_target(runner_path)
            runner_path.write_text(runner, encoding="utf-8")

            started = time.monotonic()
            output = run_sandboxed(runner_path, [], timeout=SANDBOX_TIMEOUT_SECONDS)
            elapsed_ms = int((time.monotonic() - started) * 1000)

        if "TOOL_TESTS_PASSED" not in output:
            raise RuntimeError("Sandbox tests failed: " + output[:2000])

        return {"output": output[:1000], "duration_ms": elapsed_ms}

    # ---------------------------------------------------------
    # 9. MANIFEST MANAGEMENT
    # ---------------------------------------------------------

    @staticmethod
    def _read_manifest() -> dict:
        if not MANIFEST_FILE.exists():
            return {"version": 1, "tools": []}
        assert_safe_write_target(MANIFEST_FILE)
        try:
            return json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
        except Exception:
            return {"version": 1, "tools": []}

    @staticmethod
    def _write_manifest(manifest: dict) -> None:
        assert_safe_write_target(MANIFEST_FILE.parent)
        MANIFEST_FILE.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=MANIFEST_FILE.parent, prefix=MANIFEST_FILE.name + ".", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                assert_safe_write_target(temporary)
                json.dump(manifest, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            if os.name != "nt":
                temporary.chmod(0o600)
            assert_safe_write_target(MANIFEST_FILE)
            os.replace(temporary, MANIFEST_FILE)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @staticmethod
    def _record_staged_tool(name: str, capability: str, specification: dict, destination: Path, verification: dict, file_hash: str) -> dict:
        with _MANIFEST_LOCK:
            manifest = ToolMaker._read_manifest()
            entry = {
                "name": name,
                "version": "1.0.0",
                "status": "pending_approval",
                "requested_capability": capability,
                "specification": specification,
                "path": str(destination.relative_to(PROJECT_ROOT)) if destination.is_relative_to(PROJECT_ROOT) else str(destination),
                "sha256": file_hash,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "approval": {"required": True, "approved": False, "approved_by": None},
                "validation": {"sandbox_tests": "passed", "duration_ms": verification.get("duration_ms", 0)},
                "deployment": {"status": "not_deployed"},
            }
            manifest["tools"] = [t for t in manifest["tools"] if t.get("name") != name]
            manifest["tools"].append(entry)
            ToolMaker._write_manifest(manifest)
            return entry

    # ---------------------------------------------------------
    # 10. HUMAN APPROVAL AND DEPLOYMENT (ඔබ ඉල්ලූ RETRY LOGIC එක සහිතව)
    # ---------------------------------------------------------

    def approve(self, name: str, version: str, approver_id: str) -> Any:
        if not name or not version or not approver_id:
            raise ValueError("Tool name, version and approver ID are required.")

        proposal = self.get_proposal(name, version)
        if proposal is None:
            raise FileNotFoundError("The requested staged proposal does not exist.")

        # ඔබ ඉල්ලූ වෙනස්කම 1: pending_approval සහ approved දෙකටම retry කිරීමට ඉඩ දීම
        if proposal.get("status") not in {"pending_approval", "approved"}:
            raise RuntimeError("This proposal is not eligible for deployment.")

        if proposal.get("deployment", {}).get("status") == "deployed":
            return "Tool is already deployed."

        relative_path = Path(proposal["path"])
        staged_path = (PROJECT_ROOT / relative_path).resolve() if not relative_path.is_absolute() else relative_path.resolve()

        if not staged_path.is_file():
            raise FileNotFoundError("Staged tool file no longer exists.")

        # ඔබ ඉල්ලූ වෙනස්කම 2: අලුත් Approval එකක් verify කිරීම හෝ කලින් approver ට retry කිරීමට අවසර දීම
        if proposal.get("status") == "pending_approval":
            if self._verify_approval is not None:
                authorized = self._verify_approval(name, version, approver_id)
                if authorized is not True:
                    raise PermissionError("Approval was not verified. Deployment blocked.")
            self._update_proposal(
                name,
                version,
                {
                    "status": "approved",
                    "approval": {
                        "required": True,
                        "approved": True,
                        "approved_by": approver_id,
                        "approved_at": datetime.now(timezone.utc).isoformat(),
                    },
                },
            )
        else:
            previous_approver = proposal.get("approval", {}).get("approved_by")
            if previous_approver != approver_id:
                raise PermissionError("Only the original approver can retry this deployment.")

        # Deployment Callback ක්‍රියාත්මක කිරීම
        if self._deploy_staged_tool is not None:
            deployment_result = self._deploy_staged_tool(staged_path, proposal)
        else:
            deployment_result = "Tool staged and verified."

        self._update_proposal(
            name,
            version,
            {
                "status": "deployed",
                "deployment": {
                    "status": "deployed",
                    "deployed_at": datetime.now(timezone.utc).isoformat(),
                },
            },
        )
        return deployment_result

    def get_proposal(self, name: str, version: str) -> dict | None:
        with _MANIFEST_LOCK:
            manifest = self._read_manifest()
            for item in manifest["tools"]:
                if isinstance(item, dict) and item.get("name") == name:
                    return item
        return None

    def _update_proposal(self, name: str, version: str, updates: dict) -> None:
        with _MANIFEST_LOCK:
            manifest = self._read_manifest()
            for item in manifest["tools"]:
                if isinstance(item, dict) and item.get("name") == name:
                    item.update(updates)
                    self._write_manifest(manifest)
                    return

    @staticmethod
    def _sha256(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(64 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()


def _mock_parameters(schema: dict) -> dict:
    properties = schema.get("properties", {})
    parameters = {}
    for name, definition in properties.items():
        kind = str(definition.get("type", "STRING")).upper()
        if kind == "STRING":
            parameters[name] = "test"
        elif kind in {"INTEGER", "INT"}:
            parameters[name] = 1
        elif kind in {"NUMBER", "FLOAT"}:
            parameters[name] = 1.0
        elif kind in {"BOOLEAN", "BOOL"}:
            parameters[name] = True
        else:
            parameters[name] = {}
    return parameters
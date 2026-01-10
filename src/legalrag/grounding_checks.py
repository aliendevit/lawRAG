from typing import Dict, Any

def enforce_citations(irac: Dict[str, Any]) -> bool:
    def has_cits(stmt: Dict[str, Any]) -> bool:
        return bool(stmt.get("citations"))

    if not irac:
        return False
    for item in irac.get("rule", []):
        if not has_cits(item):
            return False
    for item in irac.get("application", []):
        if not has_cits(item):
            return False
    if not has_cits(irac.get("conclusion", {})):
        return False
    return True



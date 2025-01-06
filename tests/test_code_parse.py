from cognitive_base.utils.code_parse import extract_from_ast

def test_extract_from_ast_with_import():
    # Test code that imports math and uses math.pi
    code = """
import math

def use_pi():
    return math.pi
"""
    functions, imports, deps, modules = extract_from_ast(code)
    
    assert 'math' in modules
    assert len(imports) == 1
    assert 'import math' in imports[0]
    assert len(functions) == 1
    assert functions[0]['name'] == 'use_pi'

def test_extract_from_ast_simple_import():
    # Minimal test case
    code = "import math"
    functions, imports, deps, modules = extract_from_ast(code)
    
    assert 'math' in modules
    assert len(imports) == 1
    assert 'import math' in imports[0]
    assert len(functions) == 0

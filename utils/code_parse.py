import ast
import logging
import astunparse
import builtins
import copy

from . import load_json

logger = logging.getLogger("logger")


def add_parent_references(node):
    """
    Recursively add parent references to each node in the AST.
    so that 

    Args:
        node (ast.AST): The root AST node to start adding parent references from.
    """
    for child in ast.iter_child_nodes(node):
        child.parent = node
        add_parent_references(child)


def extract_info_from_fn(node, dependencies, functions):
    """
    Extracts information from a function node in an abstract syntax tree (AST).

    This function walks through the given AST node to identify function calls and
    collects custom function names (excluding built-in functions) into the dependencies set.
    It also gathers detailed information about the function node and appends it to the functions list.

    Args:
        node (ast.FunctionDef): The AST node representing the function definition.
        dependencies (set): A set to store the names of custom functions called within the function.
        functions (list): A list to store information about the function, including its name, node, body, parameters, and whether it has a parent node.

    Returns:
        None
    """
    for inner_node in ast.walk(node):
        # Check if the node is a call to a function
        if isinstance(inner_node, ast.Call) and isinstance(inner_node.func, ast.Name):
            if inner_node.func.id not in set(dir(builtins)):
                # Add the custom function name to the list for this function
                dependencies.add(inner_node.func.id)
    fn_info = {
        "name": node.name,
        "node": node,
        "body": astunparse.unparse(node),
        "params": node.args.args,
        "no_parent": isinstance(node.parent, ast.Module),
    }
    functions.append(fn_info)


def extract_info_from_imports(node, imported_fns, import_statements, code, all_imported_modules):
    """
    Extracts information from import statements in the given AST node.

    Args:
        node (ast.AST): The AST node to analyze.
        imported_fns (list): A list to store the names of imported functions or modules.
        import_statements (list): A list to store the source code segments of import statements.
        code (str): The source code from which the AST node was generated.
        all_imported_modules (set): A set to store all imported modules.

    Returns:
        None
    """
    if isinstance(node, ast.ImportFrom):
        for alias in node.names:
            imported_fns.append(alias.name)
    import_statements.append(ast.get_source_segment(code, node))
    imported_modules = visit_imports(node)
    all_imported_modules.update(imported_modules)


def extract_from_ast(code):
    """
    Extract functions, import statements, and dependencies from the given code.

    Args:
        code (str): The code to extract information from.
        check_imports (bool, optional): Whether to check if imported modules are whitelisted. Defaults to True.

    Returns:
        tuple: A tuple containing a list of functions, a list of import statements, and a set of dependencies.

    Raises:
        Exception: If the code cannot be parsed into an AST.
    """
    try:
        parsed_ast = ast.parse(code)
    except Exception as e:
        err_msg = f'could not parse code to AST, check syntax and try again. error: {str(e)}, {type(e).__name__}\n'
        raise Exception(err_msg)

    functions = []
    import_statements = []
    dependencies = set()
    imported_fns = []
    all_imported_modules = set()

    add_parent_references(parsed_ast)
    for node in ast.walk(parsed_ast):
        if isinstance(node, ast.FunctionDef):
            extract_info_from_fn(node, dependencies, functions)
        elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            extract_info_from_imports(node, imported_fns, import_statements, code, all_imported_modules)

    for fn_name in [function['name'] for function in functions] + imported_fns:
        # allow for recursion, inner functions
        dependencies.discard(fn_name)

    return functions, import_statements, dependencies, all_imported_modules

whitelist_modules_path = "cognitive_base/utils/whitelist_modules.json"
whitelist_modules = frozenset(load_json(whitelist_modules_path))

def assert_modules_in_whitelist(imported_modules):
    """
    Asserts that all modules in the imported_modules list are present in the whitelist_modules.

    Parameters:
    imported_modules (list): A list of module names (strings) to be checked against the whitelist.

    Raises:
    AssertionError: If any module in imported_modules is not found in whitelist_modules, an AssertionError is raised with an appropriate error message.
    """
    err_msg = "Error: module {module} not in whitelist. try again without this module\n"
    for module in imported_modules:
        assert (module in whitelist_modules), err_msg.format(module=module)


def append_dependencies(parsed_result, fn_str_map_list):
    """
    Append dependencies to the code based on the function string mappings.

    This function modifies the parsed_result dictionary to include the full code
    with dependencies resolved and appended in the correct order.

    Args:
        parsed_result (dict): The parsed result dictionary containing program code and dependencies.
        fn_str_map_list (list): A list of dictionaries mapping function names to their code and dependencies.

    Returns:
        str: The full code with dependencies resolved and appended.
    """
    parsed_result['dependency_used'] = False
    dependencies = copy.deepcopy(parsed_result['dependencies'])
    visited = set()
    dependent_code = []
    while dependencies:
        print('building dependencies of fns ... \n')
        call_fn = dependencies.pop(0)
        if call_fn not in visited:
            for fn_str_map in fn_str_map_list:
                if call_fn in fn_str_map:
                    dependent_code.append(fn_str_map[call_fn]['code'])
                    parsed_result['dependency_used'] = True
                    if 'dependencies' in fn_str_map[call_fn]:
                        dependencies.extend(fn_str_map[call_fn]['dependencies'])
                    visited.add(call_fn)
                    break
            else:
                logger.warning(f'\n{call_fn} is a dependency not in entries\n')
                # print(f'\n{call_fn} is a dependency not in entries\n')
    dependent_code.reverse()
    dependent_code.append(parsed_result['program_code'])
    full_code = "\n\n".join(dependent_code)

    logger.info(f'full code to be executed: \n {full_code}\n')
    parsed_result['full_code'] = full_code

    return full_code


def visit_imports(node):
    """
    Visit an AST node and collect module names from import statements.

    Args:
        node (ast.AST): The AST node to visit.

    Returns:
        set: A set of module names imported in the node.
    """
    modules = []
    if isinstance(node, ast.Import):
        for alias in node.names:
            modules.append(alias.name)
    elif isinstance(node, ast.ImportFrom):
        modules.append(node.module)
    return set(modules)


def get_fn_name(code: str) -> str:
    """
    Extract the function name from the given code string.

    Args:
        code (str): The code string to extract the function name from.

    Returns:
        str: The name of the first function defined in the code.

    Raises:
        ValueError: If the AST parsing fails.
        AssertionError: If no function name is found.
    """
    try:
        parsed_ast = ast.parse(code)
    except Exception:
        raise ValueError('ast parse fail')
    fn_name = ''
    for node in ast.walk(parsed_ast):
        if isinstance(node, ast.FunctionDef):
            fn_name = node.name
    assert fn_name
    return fn_name

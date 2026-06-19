import ast
import numpy as np

def safe_formula(expr: str, xx, params=None):
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid formula syntax: {expr!r}") from e

    binops = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.Pow: lambda a, b: a ** b,
    }
    unops = {
        ast.UAdd: lambda a: +a,
        ast.USub: lambda a: -a,
    }

    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)

        if isinstance(node, ast.Constant):  # Py3.8+
            if isinstance(node.value, (int, float, np.number)):
                return float(node.value) if isinstance(node.value, np.number) else node.value
            raise ValueError(f"Unsupported constant: {node.value!r}")

        if isinstance(node, ast.Name):
            if node.id == "xx":
                return xx
            if node.id == "params":
                raise ValueError("Use params[<int>] indexing, not bare 'params'.")
            raise ValueError(f"Unknown name: {node.id!r}")

        if isinstance(node, ast.Subscript):
            if not (isinstance(node.value, ast.Name) and node.value.id == "params"):
                raise ValueError("Only params[<int>] indexing is allowed.")

            sl = node.slice
            if isinstance(sl, ast.Constant) and isinstance(sl.value, (int, np.integer)):
                idx = int(sl.value)
            else:
                raise ValueError("params index must be an integer literal like params[0].")

            try:
                return params[idx]
            except Exception as e:
                raise ValueError(f"Invalid params index {idx}") from e

        if isinstance(node, ast.BinOp) and type(node.op) in binops:
            f = binops[type(node.op)]
            return f(ev(node.left), ev(node.right))

        if isinstance(node, ast.UnaryOp) and type(node.op) in unops:
            f = unops[type(node.op)]
            return f(ev(node.operand))

        raise ValueError(f"Unsupported expression element: {type(node).__name__}")

    return ev(tree)
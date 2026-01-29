import re
import math
import numpy as np   
from collections import defaultdict, OrderedDict

# ---------- Parser ----------
def split_args(expr):
    """Split top-level comma-separated args (handles nested parentheses)."""
    depth = 0
    args = []
    start = 0
    for i, ch in enumerate(expr):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == ',' and depth == 0:
            args.append(expr[start:i].strip())
            start = i + 1
    args.append(expr[start:].strip())
    return args

def parse_expr(s):
    s = s.strip()
    if '(' not in s:  # number or variable
        try:
            return {'type':'num', 'value': float(s)}
        except ValueError:
            return {'type':'var', 'name': s}
    m = re.match(r'^([a-zA-Z_]+)\s*\(', s)
    if not m:
        raise ValueError("Cannot parse: " + s)
    op = m.group(1)
    assert s.endswith(')'), "Malformed expression (missing )): " + s
    inner = s[len(op)+1:-1]
    parts = split_args(inner)
    children = [parse_expr(p) for p in parts]
    return {'type':'op', 'op':op, 'args': children}

# ---------- AST printing ----------
def ast_pretty(node, indent=0):
    pad = '  ' * indent
    if node['type'] == 'num':
        return pad + f"Num({node['value']})"
    if node['type'] == 'var':
        return pad + f"Var({node['name']})"
    s = pad + f"Op({node['op']})"
    for c in node['args']:
        s += '\n' + ast_pretty(c, indent+1)
    return s

# ---------- Constant folding ----------
def fold_constants(node):
    if node['type'] in ('num','var'):
        return node
    args = [fold_constants(c) for c in node['args']]
    op = node['op']
    if all(a['type']=='num' for a in args):
        vals = [a['value'] for a in args]
        try:
            if op == 'add': return {'type':'num', 'value': vals[0] + vals[1]}
            if op == 'sub': return {'type':'num', 'value': vals[0] - vals[1]}
            if op == 'mul': return {'type':'num', 'value': vals[0] * vals[1]}
            if op == 'div': return {'type':'num', 'value': vals[0] / vals[1]}
            if op == 'sqrt': return {'type':'num', 'value': float(np.sqrt(np.abs(vals[0])))}
            if op == 'log': return {'type':'num', 'value': float(math.log(vals[0]))}
            if op == 'abs': return {'type':'num', 'value': float(abs(vals[0]))}
            if op == 'neg': return {'type':'num', 'value': -vals[0]}
            if op == 'inv': return {'type':'num', 'value': 1.0/vals[0]}
            if op == 'max': return {'type':'num', 'value': max(vals)}
            if op == 'min': return {'type':'num', 'value': min(vals)}
            if op == 'sin': return {'type':'num', 'value': float(math.sin(vals[0]))}
            if op == 'cos': return {'type':'num', 'value': float(math.cos(vals[0]))}
            if op == 'tan': return {'type':'num', 'value': float(math.tan(vals[0]))}
        except Exception:
            pass
    return {'type':'op', 'op':op, 'args': args}

# ---------- Utility: flatten mul ----------
def flatten_mul(node):
    if node['type']=='op' and node['op']=='mul':
        facs = []
        for c in node['args']:
            facs.extend(flatten_mul(c))
        return facs
    else:
        return [node]

# ---------- Distribute mul over add ----------
def distribute_mul(node):
    if node['type'] in ('num','var'):
        return node
    node = {'type':'op','op':node['op'],'args':[distribute_mul(c) for c in node['args']]}
    if node['op'] != 'mul':
        return node
    factors = flatten_mul(node)
    num_factor = 1.0
    remaining = []
    for f in factors:
        if f['type']=='num':
            num_factor *= f['value']
        else:
            remaining.append(f)
    for i, f in enumerate(remaining):
        if f['type']=='op' and f['op']=='add':
            a,b = f['args']
            other = remaining[:i]+remaining[i+1:]
            def make_mul_with(x):
                mul_factors = []
                if abs(num_factor-1.0)>1e-12:
                    mul_factors.append({'type':'num','value':num_factor})
                mul_factors.extend(other)
                mul_factors.append(x)
                if len(mul_factors)==1:
                    return mul_factors[0]
                out = mul_factors[0]
                for ff in mul_factors[1:]:
                    out={'type':'op','op':'mul','args':[out,ff]}
                return out
            return {'type':'op','op':'add','args':[distribute_mul(make_mul_with(a)), distribute_mul(make_mul_with(b))]}
    all_facs=[]
    if abs(num_factor-1.0)>1e-12:
        all_facs.append({'type':'num','value':num_factor})
    all_facs.extend(remaining)
    if len(all_facs)==0:
        return {'type':'num','value':num_factor}
    if len(all_facs)==1:
        return all_facs[0]
    out = all_facs[0]
    for ff in all_facs[1:]:
        out={'type':'op','op':'mul','args':[out,ff]}
    return out

# ---------- Detect (a-b)(a+b) ----------
def detect_difference_of_squares(node):
    if node['type']=='op' and node['op']=='mul':
        factors = flatten_mul(node)
        n=len(factors)
        for i in range(n):
            for j in range(i+1,n):
                a=factors[i]; b=factors[j]
                if a['type']=='op' and b['type']=='op':
                    if a['op']=='sub' and b['op']=='add':
                        A1,B1=a['args']; A2,B2=b['args']
                        if ast_to_infix(A1)==ast_to_infix(A2) and ast_to_infix(B1)==ast_to_infix(B2):
                            new_factor={'type':'op','op':'sub','args':[
                                {'type':'op','op':'mul','args':[A1,A1]},
                                {'type':'op','op':'mul','args':[B1,B1]}
                            ]}
                            new_factors=[f for k,f in enumerate(factors) if k not in (i,j)]
                            new_factors.append(new_factor)
                            if len(new_factors)==1:
                                return detect_difference_of_squares(new_factors[0])
                            out=new_factors[0]
                            for ff in new_factors[1:]:
                                out={'type':'op','op':'mul','args':[out,ff]}
                            return detect_difference_of_squares(out)
                    if a['op']=='add' and b['op']=='sub':
                        A1,B1=b['args']; A2,B2=a['args']
                        if ast_to_infix(A1)==ast_to_infix(A2) and ast_to_infix(B1)==ast_to_infix(B2):
                            new_factor={'type':'op','op':'sub','args':[
                                {'type':'op','op':'mul','args':[A1,A1]},
                                {'type':'op','op':'mul','args':[B1,B1]}
                            ]}
                            new_factors=[f for k,f in enumerate(factors) if k not in (i,j)]
                            new_factors.append(new_factor)
                            if len(new_factors)==1:
                                return detect_difference_of_squares(new_factors[0])
                            out=new_factors[0]
                            for ff in new_factors[1:]:
                                out={'type':'op','op':'mul','args':[out,ff]}
                            return detect_difference_of_squares(out)
    if node['type']=='op':
        return {'type':'op','op':node['op'],'args':[detect_difference_of_squares(c) for c in node['args']]}
    return node

# ---------- AST -> infix ----------
def ast_to_infix(node):
    if node['type']=='num':
        v=node['value']
        if abs(v)<1e-12: v=0.0
        return format(v,'.6g')
    if node['type']=='var':
        return node['name']
    op=node['op']; a=node['args']
    if op=='add': return f"({ast_to_infix(a[0])} + {ast_to_infix(a[1])})"
    if op=='sub': return f"({ast_to_infix(a[0])} - {ast_to_infix(a[1])})"
    if op=='mul':
        facs=flatten_mul(node)
        if len(facs)==2 and facs[0]['type']=='num':
            return f"({format(facs[0]['value'],'.6g')}*{ast_to_infix(facs[1])})"
        return "("+" * ".join(ast_to_infix(f) for f in facs)+")"
    if op == 'sqrt': return f"np.sqrt({ast_to_infix(a[0])})"
    if op == 'log':  return f"np.log({ast_to_infix(a[0])})"
    if op == 'abs':  return f"np.abs({ast_to_infix(a[0])})"
    if op == 'neg':  return f"(-{ast_to_infix(a[0])})"
    if op == 'inv':  return f"(1.0/({ast_to_infix(a[0])}))"
    if op == 'max':  return f"np.maximum({', '.join(ast_to_infix(c) for c in a)})"
    if op == 'min':  return f"np.minimum({', '.join(ast_to_infix(c) for c in a)})"
    if op == 'sin':  return f"np.sin({ast_to_infix(a[0])})"
    if op == 'cos':  return f"np.cos({ast_to_infix(a[0])})"
    if op == 'tan':  return f"np.tan({ast_to_infix(a[0])})"

    return f"{op}(" + ", ".join(ast_to_infix(c) for c in a) + ")"

# ---------- Collect additive terms ----------
def get_terms(node):
    if node['type']=='num':
        return {'__const__': node['value']}
    if node['type']=='op' and node['op']=='add':
        left=get_terms(node['args'][0]); right=get_terms(node['args'][1])
        out=defaultdict(float)
        for k,v in left.items(): out[k]+=v
        for k,v in right.items(): out[k]+=v
        return dict(out)
    if node['type']=='op' and node['op']=='sub':
        left=get_terms(node['args'][0]); right=get_terms(node['args'][1])
        out=defaultdict(float)
        for k,v in left.items(): out[k]+=v
        for k,v in right.items(): out[k]-=v
        return dict(out)
    key=ast_to_infix(node)
    return {key:1.0}

# ---------- Analyzer ----------
def analyze(expr_str, verbose=True, save_path=None):
    output_lines=[]
    def log(msg=""):
        print(msg); output_lines.append(str(msg))
    log("ORIGINAL:"); log(expr_str); log()
    ast=parse_expr(expr_str)
    if verbose: log("PARSED AST:"); log(ast_pretty(ast)); log()
    folded=fold_constants(ast)
    if verbose: log("AFTER CONSTANT FOLDING:"); log(ast_pretty(folded)); log("=> "+ast_to_infix(folded)); log()
    distributed=distribute_mul(folded)
    if verbose: log("AFTER DISTRIBUTING NUMERIC MULTIPLICATION:"); log(ast_pretty(distributed)); log("=> "+ast_to_infix(distributed)); log()
    dos=detect_difference_of_squares(distributed)
    if verbose: log("AFTER DETECTING (a-b)*(a+b) PATTERNS:"); log(ast_pretty(dos)); log("=> "+ast_to_infix(dos)); log()
    terms_map=get_terms(dos)
    terms_map={k:float(v) for k,v in terms_map.items() if abs(v)>1e-12}
    if verbose: log("COLLECTED TERMS:"); [log(f"  {repr(k)} : {v}") for k,v in terms_map.items()]; log()
    ordered=[]; const_val=terms_map.pop('__const__',0.0) if '__const__' in terms_map else 0.0
    for k,v in terms_map.items():
        if abs(v-1.0)<1e-12: ordered.append((k,None))
        else: ordered.append((k,v))
    if abs(const_val)>1e-12: ordered.append(("CONST",const_val))
    log("FINAL ENUMERATED TERMS:")
    terms_out=[]
    for i,(term_key,coeff) in enumerate(ordered,1):
        if term_key=="CONST": term_str=f"{coeff:.6g}"
        else: term_str=f"{term_key}" if coeff is None else f"{coeff:.6g} * {term_key}"
        log(f"Term {i}: {term_str}"); terms_out.append(term_str)
    log()
    formula_parts=[]
    for term_key,coeff in ordered:
        if term_key=="CONST": formula_parts.append(f"{coeff:.6g}")
        else: formula_parts.append(f"{term_key}" if coeff is None else f"{coeff:.6g}*{term_key}")
    final_formula=" + ".join(formula_parts).replace('+ -','- ')
    log("FINAL FORMULA:"); log(final_formula)
    if save_path: open(save_path,"w").write("\n".join(output_lines)); print(f"\nOutput saved to {save_path}")
    return {'ast':ast,'folded':folded,'distributed':distributed,'dos':dos,'terms_map':terms_map,
            'const':const_val,'terms_out':terms_out,'final_formula':final_formula}

# ---------- Example ----------
if __name__=="__main__":
    expr="add(sin(X0), mul(cos(X1), -2))"
    analyze(expr, verbose=True)

#!/usr/bin/env python3
"""Analisis AST de nombres huerfanos.

Busca variables usadas (Name en contexto Load) que no esten definidas en
su scope ni en ninguno de los scopes que lo contienen. Sirve para cazar
el error tipico de una refactorizacion: se borra o renombra una variable
y queda una referencia suelta que `py_compile` no detecta, porque
sintacticamente el archivo sigue siendo valido y el NameError solo
aparece en runtime, en la rama que nadie probo.

Implementa scoping lexico de verdad: cada funcion, lambda y
comprehension es un scope propio con sus parametros, y las referencias
se resuelven subiendo por la cadena de scopes hasta el modulo y los
builtins.

Uso:
    python3 nombres_huerfanos.py archivo.py [archivo2.py ...]
    python3 nombres_huerfanos.py --json archivo.py   # para diffear

Salida: una linea por hallazgo. Exit code 1 si encontro algo.
"""
from __future__ import annotations

import argparse
import ast
import builtins
import json
import sys

BUILTINS = frozenset(dir(builtins)) | {"__file__", "__name__", "__doc__"}


class Scope:
    def __init__(self, padre=None, nombre="<module>"):
        self.padre = padre
        self.nombre = nombre
        self.bindings: set[str] = set()
        self.globals_declarados: set[str] = set()

    def resolver(self, nombre: str) -> bool:
        s = self
        while s is not None:
            if nombre in s.bindings:
                return True
            s = s.padre
        return nombre in BUILTINS


def _nombres_de_target(target) -> set[str]:
    """Nombres ligados por un target de asignacion / for / with / comp."""
    out = set()
    for n in ast.walk(target):
        if isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            out.add(n.id)
    return out


class Analizador:
    """Dos pasadas por scope: primero se recolectan TODOS los bindings
    del scope (Python permite usar un nombre definido mas abajo dentro
    de una funcion, porque la resolucion es en runtime), despues se
    verifican los Loads."""

    def __init__(self, path: str):
        self.path = path
        self.hallazgos: list[tuple[str, str, int]] = []

    # ---------- recoleccion ----------
    def _bindings_de_cuerpo(self, cuerpo, scope: Scope) -> None:
        """Recorre un cuerpo SIN entrar en scopes anidados y registra
        todo lo que liga nombres en `scope`."""
        pila = list(cuerpo)
        while pila:
            n = pila.pop()
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                scope.bindings.add(n.name)
                continue  # su cuerpo es otro scope
            if isinstance(n, ast.ClassDef):
                scope.bindings.add(n.name)
                continue
            if isinstance(n, ast.Lambda):
                continue
            if isinstance(n, (ast.ListComp, ast.SetComp,
                              ast.DictComp, ast.GeneratorExp)):
                # scope propio, pero el primer iterable se evalua afuera
                pila.append(n.generators[0].iter)
                continue
            if isinstance(n, (ast.Import, ast.ImportFrom)):
                for a in n.names:
                    scope.bindings.add((a.asname or a.name).split(".")[0])
            elif isinstance(n, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
                tgts = n.targets if isinstance(n, ast.Assign) else [n.target]
                for t in tgts:
                    scope.bindings |= _nombres_de_target(t)
            elif isinstance(n, (ast.For, ast.AsyncFor)):
                scope.bindings |= _nombres_de_target(n.target)
            elif isinstance(n, (ast.With, ast.AsyncWith)):
                for item in n.items:
                    if item.optional_vars is not None:
                        scope.bindings |= _nombres_de_target(
                            item.optional_vars)
            elif isinstance(n, ast.ExceptHandler):
                if n.name:
                    scope.bindings.add(n.name)
            elif isinstance(n, (ast.Global, ast.Nonlocal)):
                scope.bindings.update(n.names)
                scope.globals_declarados.update(n.names)
            elif isinstance(n, ast.NamedExpr):
                scope.bindings |= _nombres_de_target(n.target)
            elif isinstance(n, (ast.Match,)):
                for t in ast.walk(n):
                    if isinstance(t, ast.MatchAs) and t.name:
                        scope.bindings.add(t.name)
                    if isinstance(t, ast.MatchStar) and t.name:
                        scope.bindings.add(t.name)
            # descender
            for hijo in ast.iter_child_nodes(n):
                pila.append(hijo)

    # ---------- verificacion ----------
    def _loads_de_cuerpo(self, cuerpo):
        """Name-Loads directos del scope, sin entrar en scopes anidados."""
        pila = list(cuerpo)
        anidados = []
        while pila:
            n = pila.pop()
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef,
                              ast.Lambda, ast.ClassDef)):
                anidados.append(n)
                continue
            if isinstance(n, (ast.ListComp, ast.SetComp,
                              ast.DictComp, ast.GeneratorExp)):
                anidados.append(n)
                continue
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                yield n
            for hijo in ast.iter_child_nodes(n):
                pila.append(hijo)
        self._pendientes = anidados

    def procesar(self, nodo_cuerpo, scope: Scope, etiqueta: str):
        self._bindings_de_cuerpo(nodo_cuerpo, scope)
        loads = list(self._loads_de_cuerpo(nodo_cuerpo))
        anidados = self._pendientes
        for nm in loads:
            if not scope.resolver(nm.id):
                self.hallazgos.append((etiqueta, nm.id, nm.lineno))
        for sub in anidados:
            self.procesar_scope_anidado(sub, scope, etiqueta)

    def procesar_scope_anidado(self, nodo, padre: Scope, etiqueta: str):
        if isinstance(nodo, ast.ClassDef):
            s = Scope(padre, nodo.name)
            self.procesar(nodo.body, s, f"{etiqueta}.{nodo.name}")
            return
        if isinstance(nodo, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.Lambda)):
            nombre = getattr(nodo, "name", "<lambda>")
            s = Scope(padre, nombre)
            args = nodo.args
            for a in (list(args.posonlyargs) + list(args.args)
                      + list(args.kwonlyargs)):
                s.bindings.add(a.arg)
            if args.vararg:
                s.bindings.add(args.vararg.arg)
            if args.kwarg:
                s.bindings.add(args.kwarg.arg)
            cuerpo = nodo.body if isinstance(nodo.body, list) else [
                ast.Expr(value=nodo.body)]
            self.procesar(cuerpo, s, f"{etiqueta}.{nombre}"
                          if etiqueta != "<module>" else nombre)
            return
        # comprehensions
        s = Scope(padre, "<comp>")
        for gen in nodo.generators:
            s.bindings |= _nombres_de_target(gen.target)
        partes = []
        for gen in nodo.generators:
            partes.extend(gen.ifs)
        if isinstance(nodo, ast.DictComp):
            partes += [nodo.key, nodo.value]
        else:
            partes.append(nodo.elt)
        partes += [g.iter for g in nodo.generators[1:]]
        self.procesar([ast.Expr(value=p) for p in partes], s,
                      f"{etiqueta}.<comp>")


def analizar(path: str):
    tree = ast.parse(open(path, encoding="utf-8").read(), filename=path)
    a = Analizador(path)
    a.procesar(tree.body, Scope(None, "<module>"), "<module>")
    return sorted(set(a.hallazgos), key=lambda h: (h[2], h[1]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("archivos", nargs="+")
    ap.add_argument("--json", action="store_true",
                    help="Salida JSON, para comparar antes/despues.")
    args = ap.parse_args()

    todo = {}
    for path in args.archivos:
        todo[path] = analizar(path)

    if args.json:
        print(json.dumps(
            {k: [[a, b, c] for a, b, c in v] for k, v in todo.items()},
            indent=1, ensure_ascii=False, sort_keys=True))
    else:
        for path, hs in todo.items():
            if not hs:
                print(f"{path}: sin nombres huerfanos")
                continue
            print(f"{path}: {len(hs)} nombres huerfanos")
            for etiqueta, nombre, ln in hs:
                print(f"  {path}:{ln}  {etiqueta}  -> '{nombre}'")
    return 1 if any(todo.values()) else 0


if __name__ == "__main__":
    sys.exit(main())

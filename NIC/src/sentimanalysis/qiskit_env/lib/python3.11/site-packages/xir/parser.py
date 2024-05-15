"""This module contains the :class:`xir.Transformer` class and the XIR parser."""
import math
from decimal import Decimal
from itertools import chain

import lark
from lark import v_args

from .decimal_complex import DecimalComplex
from .program import Declaration, ObservableFactor, ObservableStmt, Program, Statement
from .utils import simplify_math


class Transformer(lark.Transformer):
    """Transformer for processing the Lark parse tree.

    Transformers visit each node of the tree, and run the appropriate method on it according to the
    node's data. All method names mirror the corresponding symbols from the grammar.

    Keyword args:
        eval_pi (bool): Whether pi should be evaluated and stored as a float
            instead of symbolically as a string. Defaults to ``False``.
        use_floats (bool): Whether floats and complex types are returned instead of ``Decimal``
            and ``DecimalComplex`` objects. Defaults to ``True``.
    """

    def __init__(self, *args, **kwargs):
        self._eval_pi = kwargs.pop("eval_pi", False)
        self._use_floats = kwargs.pop("use_floats", True)
        self._program = Program()
        super().__init__(*args, **kwargs)

    @property
    def eval_pi(self) -> bool:  # pylint: disable=used-before-assignment
        """Reports whether pi is evaluated and stored as a float."""
        return self._eval_pi

    @property
    def use_floats(self) -> bool:
        """Reports whether floats and complex types are used."""
        return self._use_floats

    def program(self, args):
        """Root of AST containing include statements and the main circuit.

        Returns:
            Program: program containing all parsed data
        """
        # assert all stmts are handled
        assert all(a is None for a in args)
        return self._program

    def circuit(self, args):
        """Any statement that is not an include. Appends gate and output
        statements to the program.
        """
        for stmt in args:
            if isinstance(stmt, Statement):
                self._program.add_statement(stmt)

    def script_options(self, args):
        """Script-level options. Adds any options to the program."""
        for name, value in args:
            self._program.add_option(name, value)

    def constants(self, args):
        """Script-level constants. Adds any constant to the program."""
        for name, value in args:
            self._program.add_constant(name, value)

    ###############
    # basic types
    ###############

    def int_(self, n):
        """Signed integers"""
        return int(n[0])

    def uint(self, n):
        """Unsigned integers"""
        return int(n[0])

    def float_(self, d):
        """Floating point numbers"""
        return Decimal(d[0])

    def imag_(self, c):
        """Imaginary numbers"""
        return DecimalComplex("0.0", c[0])

    def bool_(self, b):
        """Boolean expressions"""
        return bool(b[0])

    def wires(self, w):
        """Tuple with wires and identifier"""
        unraveled_w = []
        for i in w:
            if isinstance(i, range):
                unraveled_w.extend(i)
            else:
                unraveled_w.append(i)
        return "wires", tuple(unraveled_w)

    def params_list(self, p):
        """Tuple with list of params and identifier"""
        return "params", list(p)

    def params_dict(self, p):
        """Tuple with dictionary of params and identifier"""
        return "params", dict(p)

    option = tuple
    array = list

    INVERSE = str
    CTRL = str

    ANGLE_L = str
    ANGLE_R = str

    def FALSE_(self, _):
        """Returns False"""
        return False

    def TRUE_(self, _):
        """Returns True"""
        return True

    #############################
    # includes
    #############################

    def include(self, args):
        """Includes an external XIR script."""
        include = "".join(map(str, args))
        self._program.add_include(include)

    def path(self, args):
        """Path to an included XIR script."""
        return str(args[0])

    #############################
    # variables and expressions
    #############################

    def var(self, v):
        """String expressions that can be substituted by values at a later stage."""
        self._program.add_variable(v[0])
        return str(v[0])

    def range_(self, args):
        """Range between two signed integers"""
        return range(int(args[0]), int(args[1]))

    def name(self, n):
        """Name of variable, gate, observable, measurement type, option, external
        file, observable, wire, mathematical operation, etc."""
        return str(n[0])

    def expr(self, args):
        """Catch-all for expressions"""
        if len(args) == 1:
            return args[0]
        # if not a single value, it's a set of string expressions
        return "".join(map(str, args))

    ##############################
    # definitions and statements
    ##############################

    @v_args(inline=True)
    def gate_def(self, name, params_list, wires, *stmts):
        """Gate definition. Starts with keyword 'gate'. Adds gate to program."""

        if not wires:
            max_wire = 0
            for stmt in stmts:
                int_wires = [w for w in stmt.wires if isinstance(w, int)]
                max_wire = max((max_wire, *int_wires))

            wires = tuple(range(max_wire + 1))
        else:
            # remove duplicate wires while maintaining order
            wires = tuple(dict.fromkeys(wires))

        params = params_list[1] if params_list else []
        self._program.add_gate(name, params, wires, stmts)

    def application_stmt(self, args):
        """Application statement. Can be either a gate statement or an output statement and is
        defined either directly in the circuit or inside a gate definition.

        Returns:
            Statement: statement with the given data
        """
        inverse = False
        ctrl_wires = set()

        while args[0] in ("inv", "ctrl"):
            a = args.pop(0)
            if a == "inv":
                inverse = not inverse
            elif a == "ctrl":
                ctrl_wires.update(args.pop(0))

        name = args.pop(0)
        if is_param(args[0]):
            if isinstance(args[0][1], list):
                params = list(map(simplify_math, args[0][1]))
                wires = args[1]
            else:  # if dict
                params = {str(k): simplify_math(v) for k, v in args[0][1].items()}
                wires = args[1]
        else:
            params = []
            wires = args[0]

        stmt_options = {
            "ctrl_wires": tuple(sorted(ctrl_wires, key=hash)),
            "inverse": inverse,
            "use_floats": self.use_floats,
        }
        return Statement(name, params, wires, **stmt_options)

    @v_args(inline=True)
    def obs_def(self, name, params, wires, statements):
        """Create an observable definition.

        Creates an observable definition from text of the form:

        .. code-block:: text

            obs my_obs(params)[0, 1]:
                1, obs_1[0];
                0.5, obs_2[1];
            end;

        Args:
            name: observable name
            params: observable params
            wires: observable wires
            statements: list of statements

        """

        if wires is None:
            max_wire = 0
            for stmt in statements:
                for factor in stmt.factors:
                    int_wires = [w for w in factor.wires if isinstance(w, int)]
                    if int_wires and max(int_wires) > max_wire:
                        max_wire = max(int_wires)

            wires = tuple(range(max_wire + 1))
        else:
            # remove duplicate wires while maintaining order
            wires = tuple(dict.fromkeys(wires))

        params = params[1] if params else []
        self._program.add_observable(name, params, wires, statements)

    def obs_stmt_list(self, stmts):
        """Observable statement list"""
        return stmts

    @v_args(inline=True)
    def obs_stmt(self, pref, factors):
        """Create an ``ObservableStmt`` from prefactor and factors."""
        return ObservableStmt(simplify_math(pref), factors, use_floats=self.use_floats)

    def obs_group(self, factors):
        """Observable Factors"""
        return factors

    @v_args(inline=True)
    def obs_factor(self, name, params, wires):
        """Create ``ObservableFactor`` from name, params and wires."""
        params = params[1] if params else []
        return ObservableFactor(name, params, wires)

    ################
    # declarations
    ################

    @v_args(inline=True)
    def gate_decl(self, name, params, wires):
        """Gate declaration. Adds declaration to program."""
        params = params[1] if params else []
        decl = Declaration(name, type_="gate", params=params, wires=wires)
        self._program.add_declaration(decl)

    @v_args(inline=True)
    def obs_decl(self, name, params, wires):
        """Observable declaration. Adds declaration to program."""
        params = params[1] if params else []
        decl = Declaration(name, type_="obs", params=params, wires=wires)
        self._program.add_declaration(decl)

    def wire_list(self, args):
        """Tuple of wires."""
        nested = (arg if isinstance(arg, range) else [arg] for arg in args)
        return tuple(chain(*nested))

    def ARBITRARY_NUM_WIRES(self, _):
        """Arbitrary number of wires."""
        return ...

    @v_args(inline=True)
    def func_decl(self, name, params):
        """Function declaration. Adds function declaration to program."""
        params = params[1] if params else []
        decl = Declaration(name, type_="func", params=params)
        self._program.add_declaration(decl)

    @v_args(inline=True)
    def out_decl(self, name, params, wires):
        """Output declaration. Adds declaration to program."""
        params = params[1] if params else []
        decl = Declaration(name, type_="out", params=params, wires=wires)
        self._program.add_declaration(decl)

    ###############
    # mathematics
    ###############

    def math_op(self, args):
        """Mathemetical operation. Adds operation to the program.

        Returns:
            str: string representation of operation
        """
        self._program.add_called_function(args[0])
        return str(args[0]) + "(" + str(args[1]) + ")"

    def add(self, args):
        """Addition operation.

        Returns:
            Union[number, str]: resulting value after applied operation or string representation
            of operation if expression contains string variables
        """
        if all(isinstance(a, (int, Decimal, DecimalComplex)) for a in args):
            return args[0] + args[1]
        return "(" + " + ".join(map(str, args)) + ")"

    def sub(self, args):
        """Subtraction operation.

        Returns:
            Union[number, str]: resulting value after applied operation or string representation
            of operation if expression contains string variables
        """
        if all(isinstance(a, (int, Decimal, DecimalComplex)) for a in args):
            return args[0] - args[1]
        return "(" + " - ".join(map(str, args)) + ")"

    def prod(self, args):
        """Product operation.

        Returns:
            Union[number, str]: resulting value after applied operation or string representation
            of operation if expression contains string variables
        """
        if all(isinstance(a, (int, Decimal, DecimalComplex)) for a in args):
            return args[0] * args[1]
        return " * ".join(map(str, args))

    def div(self, args):
        """Division operation.

        Returns:
            Union[number, str]: resulting value after applied operation or string representation
            of operation if expression contains string variables
        """
        if all(isinstance(a, (int, Decimal, DecimalComplex)) for a in args):
            # if numerator and denominator are ints, then cast numerator to
            # Decimal so that no floats are being returned
            if all(isinstance(a, int) for a in args):
                return Decimal(args[0]) / args[1]
            return args[0] / args[1]
        return " / ".join(map(str, args))

    def neg(self, args):
        """Negation operation.

        Returns:
            Union[number, str]: resulting value after applied operation or string representation
            of operation if expression contains string variables
        """
        if isinstance(args[0], (int, Decimal, DecimalComplex)):
            return -args[0]
        return "-" + str(args[0])

    def PI(self, _):
        """Mathematical constant pi.

        Returns:
            Union[Decimal, str]: value of pi or string 'PI'
        """
        return "PI" if not self._eval_pi else Decimal(str(math.pi))


def is_wire(arg):
    """Returns whether the passed argument is a tuple of wires."""
    return isinstance(arg, tuple) and arg[0] == "wires"


def is_param(arg):
    """Returns whether the passed argument is a list or dictionary of params."""
    return isinstance(arg, tuple) and arg[0] == "params"

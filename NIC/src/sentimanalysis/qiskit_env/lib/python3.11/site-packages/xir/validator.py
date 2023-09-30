"""This module contains functions for validating an XIR program"""

import re
from decimal import Decimal
from itertools import chain
from typing import MutableSet, Optional, Sequence, Union

from xir.decimal_complex import DecimalComplex
from xir.program import Declaration, ObservableStmt, Param, Program, Statement, Wire

VALID_CONSTANTS = {"PI"}


class ValidationError(Exception):
    """Exception raised when validation of an XIR program fails.

    Args:
        validation_msgs (Sequence, str): A Sequence of error messages describing which validators
            failed or a single custom message. If ``None`` then a generic error message is used.
    """

    def __init__(self, validation_msgs: Union[None, Sequence, str] = None) -> None:
        if isinstance(validation_msgs, str):
            self.message = f"XIR program is invalid: {validation_msgs}"

        elif isinstance(validation_msgs, Sequence) and validation_msgs:
            validations_str = "\n\t-> " + "\n\t-> ".join(map(str, validation_msgs))
            self.message = (
                "XIR program is invalid: the following issues have been detected:"
                f"{validations_str}"
            )

        else:  # backwards compatible with string messages
            self.message = "XIR program is invalid."

        super().__init__(self.message)


def stem(val: str) -> str:
    """Return stem of function or observable call by removing any terminal parantheses.

    As an example, passing ``"function(param)"`` returns ``"function"``.

    Args:
        val (str): function call string

    Returns:
        str: stem of input string
    """
    match = re.match(r"(\w+)\s*\(.*?\)\s*$", val)
    if match:
        return match[1]
    return val


class Validator:
    """Validator used to validate an XIR program.

    Checks if an XIR program is correctly defined. While the XIR parser and transformer catch
    syntactic errors, the validator ensures that all statements are logically valid and consistent.

    Args:
        program (Program): the program which is to be validated
        ignore_includes (bool): whether to ignore any include statements and always check that all
            statements have their corresponding declarations in the same file

    **Example:**

    The ``Validator`` class is used to create a validator object which accepts a program to
    be validated. The validator can then be run by calling the ``run()`` method. Optionally,
    the user can set the ``raise_exception`` flag, which determines whether or not to raise a
    ``ValidationError`` or return ``None``, or to simply return the error message log without
    raising an exception.

    .. code-block:: python

        valid_script = \"""
            gate rx(a)[0];
            out amplitude(state)[0..2];

            rx(0.7) | [0];
            rx(1.2) | [1];

            amplitude(state: [1, 0]) | [0..2];
        \"""

        prog = parse_script(valid_script)
        xir.Validator(prog).run()

    Optionally, the validator can be set to ignore any include statements, meaning that all gates,
    outputs, functions and observables must be declared in the same file.

    .. code-block:: python

        validator = xir.Validator(prog, ignore_includes=True)
        log = validator(raise_exception=False).run()
    """

    def __init__(self, program: Program, ignore_includes: bool = False) -> None:
        self._validation_messages = []
        self._ignore_includes = ignore_includes

        self._program = program
        self.has_includes = len(program.includes) > 0

        self._validators = {
            "constants": True,
            "declarations": True,
            "statements": True,
            "definitions": True,
        }

    def run(self, raise_exception: bool = True) -> Optional[Sequence[str]]:
        """Runs the validation checks.

        Args:
            raise_exception (bool): whether to raise a ``ValidationError``
                if any issues are found

        Returns:
            Sequence[str], None: list of potential issues iff ``raise_exception`` is ``False``
            and at least one validation error was detected

        Raises:
            ValidationError: if any issues are found and ``raise_exception`` is ``True``
        """
        # reset validation messages in case previously run
        self._validation_messages = []

        if self._validators["constants"]:
            self._check_constants()

        if self._validators["declarations"]:
            self._check_declarations()

        if self._validators["statements"]:
            self._check_statements()

        if self._validators["definitions"]:
            self._check_recursive_definitions()
            self._check_gate_definitions()
            self._check_observable_definitions()

        if self._validation_messages and raise_exception:
            raise ValidationError(self._validation_messages)

        return self._validation_messages or None

    def _check_constants(self) -> None:
        """Checks that the declared constants are valid."""
        for const in self._program.constants.keys():
            # grammar uses lowercase constants, while the program uses upper-case
            if const in map(str.lower, VALID_CONSTANTS):
                msg = f"Constant '{const}' is already defined and cannot be replaced."
                self._validation_messages.append(msg)

    def _check_declarations(self) -> None:
        """Checks that declarations are valid.

        Checks that all declarations have correct wires, labels and parameters.
        The declarations will be marked as invalid if:

            * A declaration has duplicate wire labels.
            * A declaration has duplicate parameter names.
            * A declaration has labels which are not strings.
            * A declaration has parameters which are not strings.
        """
        for decl in (d for l in self._program.declarations.values() for d in l):
            if decl.wires != ... and len(set(decl.wires)) != len(decl.wires):
                msg = f"Declaration '{decl}' has duplicate wires labels."
                self._validation_messages.append(msg)

            if len(set(decl.params)) != len(decl.params):
                msg = f"Declaration '{decl}' has duplicate parameter names."
                self._validation_messages.append(msg)

            if not all(isinstance(p, str) for p in decl.params):
                msg = f"Declaration '{decl}' has parameters which are not strings."
                self._validation_messages.append(msg)

    def _check_statements(
        self,
        statements: Optional[Sequence[Statement]] = None,
        declared_params: Optional[Sequence[str]] = None,
    ) -> None:
        """Checks that statements are valid.

        Checks that all statements have a declaration and that they are correctly applied. If no
        statements are passed, the script-level statements in the program will be used. The
        statements will be marked as invalid if:

            * A gate application statement specifies the wrong number of wires.
            * A gate application statement specifies the wrong number of parameters.
            * A gate application statement specifies the wrong parameter names.
            * A gate application statement at the script level is applied to named wires.

        Checks that modifiers such as ``ctrl`` and ``inv`` are correctly applied.
        The modifiers will be marked as invalid if:

            * An ``inv`` or ``ctrl`` modifier is applied to an output statement.
            * There are duplicate wires specified in a ``ctrl`` modifier.

        If ``declared_params`` are passed, parameters may have the containing names.
        """
        declarations = list(
            chain(self._program.declarations["gate"], self._program.declarations["out"])
        )
        declaration_names = [x.name for x in declarations]

        statements = statements or self._program.statements
        for stmt in statements:
            if stmt.name in declaration_names:
                self._check_statements_with_declarations(stmt, declarations, declaration_names)

            elif self._ignore_includes or not self.has_includes:
                msg = f"Name '{stmt.name}' has not been declared."
                self._validation_messages.append(msg)

            self._check_statement_applications(stmt, declared_params)

    def _check_statements_with_declarations(
        self, stmt: Statement, declarations: Sequence[Declaration], declaration_names: Sequence[str]
    ) -> None:
        """Checks that statements comply with their declarations."""
        idx = declaration_names.index(stmt.name)

        # check that dict-like parameters use the correct parameter names
        if isinstance(stmt.params, dict):
            if set(stmt.params.keys()) != set(declarations[idx].params):
                expected = ", ".join(declarations[idx].params)
                msg = f"Statement '{stmt}' passes the wrong parameters. Expected '{expected}'."
                self._validation_messages.append(msg)

        # check that the number of parameters agree with the declaration
        # redundant if prior error is present
        elif len(stmt.params) != len(declarations[idx].params):
            expected = len(declarations[idx].params)
            msg = f"Statement '{stmt}' has {len(stmt.params)} parameter(s). Expected {expected}."
            self._validation_messages.append(msg)

        # check that statements are applied to the correct number of wires
        if declarations[idx].wires != ... and len(stmt.wires) != len(declarations[idx].wires):
            expected = len(declarations[idx].wires)
            msg = f"Statement '{stmt}' has {len(stmt.wires)} wire(s). Expected {expected}."
            self._validation_messages.append(msg)

        # check that gates aren't applied to duplicate wires
        if len(set(stmt.wires)) != len(stmt.wires):
            msg = f"Statement '{stmt}' is applied to duplicate wires."
            self._validation_messages.append(msg)

        # check that `inv` or `ctrl` isn't applied to an output statement
        if (stmt.is_inverse or stmt.ctrl_wires) and declarations[idx].type_ == "out":
            msg = f"Statement '{stmt}' is an output statement but has 'ctrl' or 'inv' modifiers."
            self._validation_messages.append(msg)

    def _check_statement_applications(
        self, stmt: Statement, declared_params: Optional[Sequence[str]]
    ) -> None:
        """Checks that statements are valid.

        Checks the remaining conditions documented (but not implemented) by
        :func:`Validator._check_statements()`.
        """
        # check that only integer wire labels are used at the script level
        if declared_params is None and any(isinstance(wire, str) for wire in stmt.wires):
            msg = (
                f"Statement '{stmt}' is applied to named wires. Only integer wire labels are "
                "allowed at the script level."
            )
            self._validation_messages.append(msg)

        # check that control wires aren't the same as applied wires
        ctrl_stmt_wires = set(stmt.ctrl_wires) & set(stmt.wires)
        if ctrl_stmt_wires:
            msg = f"Statement '{stmt}' has control wires {ctrl_stmt_wires} which are also applied."
            self._validation_messages.append(msg)

    def _check_recursive_definitions(self) -> None:
        """Checks that a gate is not defined in terms of itself."""

        def _recursive_check(name: str, definition: Sequence[Statement], stack: MutableSet[str]):
            if name in stack:
                msg = f"Gate definition '{name}' has a circular dependency."
                self._validation_messages.append(msg)
                return

            stack.add(name)

            for stmt in definition:
                if stmt.name in self._program.gates.keys():
                    _recursive_check(stmt.name, self._program.gates[stmt.name], stack)

            stack.remove(name)

        for name, val in self._program.gates.items():
            _recursive_check(name, val, stack=set())

    def _check_gate_definitions(self) -> None:
        """Checks that gate definitions are valid.

        Definitions are invalid if:
            * A non-integer wire label is used which is not explicitly specified
              in the gate signature.
            * At least one wire label is specified in the gate signature and an
              integer wire label is used which is not explicitly specified in
              the gate signature.
            * A mixture of integer and non-integer wire labels are specified in
              the gate signature.
            * A wire label which is specified in the gate signature is not used.
            * The same name is used for wires and parameters.
            * The statements inside gate definitions are not correctly declared.
        """
        for name, statements in self._program.gates.items():
            # check that statements in definitions are valid
            declared_params = self._program.search("gate", "params", name)
            self._check_statements(statements, declared_params)

            applied_wires = list(
                chain(*map(lambda x: list(x.wires) + list(x.ctrl_wires), statements))
            )

            declared_wires = self._program.search("gate", "wires", name)

            self._shared_definition_checks(name, declared_wires, applied_wires, declared_params)

    def _check_observable_definitions(self) -> None:
        """Check that observable definitions are valid.

        Definitions are invalid if:
            * A non-integer wire label is used which is not explicitly specified
              in the observable signature.
            * At least one wire label is specified in the observable signature
              and an integer wire label is used which is not explicitly
              specified in the observable signature.
            * A mixture of integer and non-integer wire labels are specified in
              the observable signature.
            * A wire label which is specified in the gate signature is not used.
            * The same name is used for wires and parameters.
            * The statements inside observable definitions are not correctly declared.
        """
        for name, statements in self._program.observables.items():
            declared_params = self._program.search("obs", "params", name)

            declared_wires = self._program.search("obs", "wires", name)
            applied_wires = list(chain(*[x.wires for x in statements]))

            self._check_observable_statements(statements, declared_params)
            self._shared_definition_checks(name, declared_wires, applied_wires, declared_params)

    def _shared_definition_checks(
        self,
        name: str,
        declared_wires: Sequence[Wire],
        applied_wires: Sequence[Wire],
        declared_params: Sequence[Param],
    ) -> None:
        """Checks the names, wires, and parameters for both observable and gate definitions."""
        any_integer_applied_wires = any(isinstance(w, int) for w in applied_wires)
        all_string_declared_wires = all(isinstance(w, str) for w in declared_wires)

        if any_integer_applied_wires and all_string_declared_wires:
            msg = (
                f"Definition '{name}' is invalid. Only named wires can be applied when "
                "declaring named wires."
            )
            self._validation_messages.append(msg)

        if set(applied_wires) - set(declared_wires):
            applied = ", ".join(map(str, applied_wires))
            declared = ", ".join(map(str, declared_wires))
            msg = (
                f"Definition '{name}' is invalid. Applied wires [{applied}] differ "
                f"from declared wires [{declared}]."
            )
            self._validation_messages.append(msg)

        if set(declared_wires) & set(declared_params):
            msg = f"Definition '{name}' is invalid. Wire and parameter names must differ."
            self._validation_messages.append(msg)

        constants_declared = set(declared_params) & set(self._program.constants)
        if constants_declared:
            msg = (
                f"Definition '{name}' is invalid. Cannot use declared constant(s) "
                f"{constants_declared} as parameter(s)."
            )
            self._validation_messages.append(msg)

    def _check_observable_statements(
        self,
        statements: Sequence[ObservableStmt],
        declared_params: Optional[Sequence[str]] = None,
    ) -> None:
        """Check that observable statements are valid.

        The statements will be marked as invalid if:

            * There is an undeclared prefactor variable.
            * An invalid observable is used.
            * A product of observables is applied to the same wire(s).
        """
        declared_funcs = [decl.name for decl in self._program.declarations["func"]]

        for stmt in statements:
            declared_pref = (
                stmt.pref in declared_params
                or str(stmt.pref) in VALID_CONSTANTS
                or (isinstance(stmt.pref, str) and stem(stmt.pref) in declared_funcs)
                or isinstance(stmt.pref, (int, float, complex, Decimal, DecimalComplex))
            )
            if not declared_pref and (self._ignore_includes or not self.has_includes):
                msg = f"Statement '{stmt}' has an undeclared prefactor variable '{stmt.pref}'."
                self._validation_messages.append(msg)

            # check if invalid observables are applied
            words = [factor.name for factor in stmt.factors]
            wires = tuple(wire for factor in stmt.factors for wire in factor.wires)
            invalid_words = set(words) - {x.name for x in self._program.declarations["obs"]}
            if invalid_words and not (not self._ignore_includes and self.has_includes):
                msg = (
                    f"Observable statement '{stmt}' is invalid. Observable(s) "
                    f"{sorted(invalid_words)} have not been declared."
                )
                self._validation_messages.append(msg)

            # check if observable products are applied to the same wires
            if len(set(wires)) != len(wires):
                msg = (
                    f"Observable statement '{stmt}' is invalid. Products of observables cannot be "
                    "applied to the same wires."
                )
                self._validation_messages.append(msg)

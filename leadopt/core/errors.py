class LeadOptError(Exception):
    """Base exception for leadopt."""


class ChemistryError(LeadOptError):
    """Raised when a chemistry transformation or validation fails."""


class ConstraintError(LeadOptError):
    """Raised when constraints cannot be established or are violated."""


class ActionError(LeadOptError):
    """Raised when an action instance is invalid or cannot be applied."""

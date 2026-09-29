"""Service-layer errors. The api layer maps them to HTTP status codes."""


class ServiceError(Exception):
    """Base class."""


class NotFoundError(ServiceError):
    pass


class ConflictError(ServiceError):
    pass


class PermissionDeniedError(ServiceError):
    pass

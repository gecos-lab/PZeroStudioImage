"""Cancellable Qt thread-pool jobs used by the application controller."""

from __future__ import annotations

from dataclasses import dataclass
import traceback
from typing import Any, Callable

from PyQt5.QtCore import QObject, QRunnable, pyqtSignal, pyqtSlot


class TaskSignals(QObject):
    completed = pyqtSignal(object)
    failed = pyqtSignal(str, str)
    progress = pyqtSignal(int, str)
    finished = pyqtSignal()


@dataclass(slots=True)
class CancellationToken:
    cancelled: bool = False

    def cancel(self) -> None:
        self.cancelled = True


class BackgroundTask(QRunnable):
    """Execute a callable without blocking the Qt event loop."""

    def __init__(self, function: Callable[..., Any], *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self.signals = TaskSignals()
        self.token = CancellationToken()
        self.setAutoDelete(True)

    @pyqtSlot()
    def run(self) -> None:
        try:
            if self.token.cancelled:
                return
            result = self.function(*self.args, **self.kwargs)
            if not self.token.cancelled:
                self.signals.completed.emit(result)
        except Exception as exc:  # Qt must receive errors on the main thread.
            if not self.token.cancelled:
                self.signals.failed.emit(str(exc), traceback.format_exc())
        finally:
            self.signals.finished.emit()

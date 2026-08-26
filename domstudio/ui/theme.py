"""Visual language for the DOMStudio desktop interface.

The UI deliberately owns its palette instead of inheriting platform colours.  That
keeps plots, overlays, and controls readable on every supported desktop.
"""

from PyQt5.QtGui import QColor, QPalette


COLORS = {
    "window": "#0E1218",
    "surface": "#151B23",
    "surface_raised": "#1B222C",
    "surface_hover": "#222B36",
    "border": "#2A3542",
    "border_strong": "#3B4A59",
    "text": "#EEF3F7",
    "text_muted": "#91A0AE",
    "text_faint": "#697887",
    "primary": "#43D6B1",
    "primary_hover": "#62E1C0",
    "primary_pressed": "#2DBB98",
    "primary_tint": "#173B35",
    "blue": "#69A7FF",
    "warning": "#F2B866",
    "danger": "#F2767A",
    "canvas": "#090C11",
}


STYLE_SHEET = """
* {
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 12px;
    color: #EEF3F7;
    outline: none;
}

QMainWindow, QWidget#AppRoot {
    background: #0E1218;
}

QMenuBar {
    background: #111720;
    border-bottom: 1px solid #26313C;
    padding: 2px 4px;
}

QMenuBar::item {
    background: transparent;
    border-radius: 4px;
    padding: 4px 8px;
}

QMenuBar::item:selected {
    background: #222B36;
}

QMenu {
    background: #1B222C;
    border: 1px solid #33404D;
    padding: 5px;
}

QMenu::item {
    border-radius: 4px;
    padding: 6px 26px 6px 9px;
}

QMenu::item:selected {
    background: #245A50;
}

QMenu::separator {
    height: 1px;
    background: #33404D;
    margin: 4px 7px;
}

QFrame#Sidebar, QFrame#Inspector {
    background: #151B23;
    border: none;
}

QFrame#Sidebar {
    border-right: 1px solid #26313C;
}

QFrame#Inspector {
    border-left: 1px solid #26313C;
}

QFrame#Header {
    background: #111720;
    border-bottom: 1px solid #26313C;
}

QFrame#CanvasToolbar, QFrame#InspectorCard, QFrame#EmptyCard {
    background: #1B222C;
    border: 1px solid #2A3542;
    border-radius: 8px;
}

QLabel#BrandMark {
    color: #43D6B1;
    font-size: 21px;
    font-weight: 700;
    letter-spacing: 1px;
}

QLabel#BrandSub, QLabel#MutedLabel, QLabel#SectionCaption,
QLabel#CoordinateLabel, QLabel#ZoomLabel {
    color: #91A0AE;
}

QLabel#PageTitle {
    font-size: 17px;
    font-weight: 600;
}

QLabel#PanelTitle {
    font-size: 13px;
    font-weight: 600;
}

QLabel#SectionCaption {
    font-size: 10px;
    font-weight: 600;
}

QLabel#MetricValue {
    color: #43D6B1;
    font-size: 17px;
    font-weight: 600;
}

QLabel#MetricLabel {
    color: #91A0AE;
    font-size: 10px;
}

QPushButton, QToolButton {
    background: #222B36;
    border: 1px solid #33404D;
    border-radius: 6px;
    min-height: 29px;
    padding: 0 11px;
}

QPushButton:hover, QToolButton:hover {
    background: #293542;
    border-color: #435365;
}

QPushButton:pressed, QToolButton:pressed {
    background: #1C252E;
}

QPushButton:disabled, QToolButton:disabled {
    background: #191F27;
    border-color: #242D37;
    color: #61707E;
}

QPushButton#PrimaryButton {
    background: #43D6B1;
    border-color: #43D6B1;
    color: #081712;
    font-weight: 600;
}

QPushButton#PrimaryButton:hover {
    background: #62E1C0;
    border-color: #62E1C0;
}

QPushButton#PrimaryButton:pressed {
    background: #2DBB98;
}

QPushButton#GhostButton {
    background: transparent;
    border-color: #33404D;
}

QPushButton#DangerButton {
    background: transparent;
    border-color: #704347;
    color: #F39A9D;
}

QToolButton#ModeButton {
    min-width: 31px;
    padding: 0 8px;
}

QToolButton#ModeButton:checked {
    background: #173B35;
    border-color: #43D6B1;
    color: #62E1C0;
}

QToolButton#WorkflowStep {
    background: transparent;
    border: 1px solid transparent;
    border-radius: 7px;
    min-height: 46px;
    padding: 3px 10px;
    text-align: left;
    color: #AEBAC5;
}

QToolButton#WorkflowStep:hover {
    background: #1B242E;
    border-color: #25313C;
}

QToolButton#WorkflowStep:checked {
    background: #173B35;
    border-color: #2A6F61;
    color: #62E1C0;
}

QToolButton#WorkflowStep:disabled {
    color: #586674;
}

QComboBox, QSpinBox, QDoubleSpinBox, QLineEdit {
    background: #111720;
    border: 1px solid #303C49;
    border-radius: 5px;
    min-height: 29px;
    padding: 0 8px;
    selection-background-color: #2A7968;
}

QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {
    border-color: #435365;
}

QComboBox::drop-down {
    border: none;
    width: 24px;
}

QComboBox QAbstractItemView {
    background: #1B222C;
    border: 1px solid #33404D;
    selection-background-color: #245A50;
}

QSlider::groove:horizontal {
    height: 4px;
    background: #303B47;
    border-radius: 2px;
}

QSlider::handle:horizontal {
    background: #43D6B1;
    border: none;
    width: 13px;
    height: 13px;
    margin: -5px 0;
    border-radius: 6px;
}

QSlider::sub-page:horizontal {
    background: #358E7B;
    border-radius: 2px;
}

QCheckBox {
    color: #BEC8D1;
    spacing: 7px;
}

QCheckBox::indicator {
    width: 15px;
    height: 15px;
    border-radius: 3px;
    background: #111720;
    border: 1px solid #3A4856;
}

QCheckBox::indicator:checked {
    background: #43D6B1;
    border-color: #43D6B1;
}

QScrollArea, QScrollArea > QWidget > QWidget {
    background: transparent;
    border: none;
}

QScrollBar:vertical {
    background: transparent;
    width: 8px;
    margin: 2px;
}

QScrollBar::handle:vertical {
    background: #354351;
    border-radius: 3px;
    min-height: 28px;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    height: 0;
    background: transparent;
}

QProgressBar {
    background: #202934;
    border: none;
    border-radius: 2px;
    height: 4px;
    max-height: 4px;
    text-align: center;
}

QProgressBar::chunk {
    background: #43D6B1;
    border-radius: 2px;
}

QStatusBar {
    background: #111720;
    border-top: 1px solid #26313C;
    color: #91A0AE;
}

QStatusBar::item {
    border: none;
}

QToolTip {
    background: #242E39;
    border: 1px solid #405061;
    color: #EEF3F7;
    padding: 4px;
}
"""


def apply_theme(application) -> None:
    """Apply the DOMStudio palette and widget stylesheet to a QApplication."""

    application.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(COLORS["window"]))
    palette.setColor(QPalette.WindowText, QColor(COLORS["text"]))
    palette.setColor(QPalette.Base, QColor(COLORS["surface"]))
    palette.setColor(QPalette.AlternateBase, QColor(COLORS["surface_raised"]))
    palette.setColor(QPalette.ToolTipBase, QColor(COLORS["surface_raised"]))
    palette.setColor(QPalette.ToolTipText, QColor(COLORS["text"]))
    palette.setColor(QPalette.Text, QColor(COLORS["text"]))
    palette.setColor(QPalette.Button, QColor(COLORS["surface_raised"]))
    palette.setColor(QPalette.ButtonText, QColor(COLORS["text"]))
    palette.setColor(QPalette.BrightText, QColor("#FFFFFF"))
    palette.setColor(QPalette.Highlight, QColor(COLORS["primary"]))
    palette.setColor(QPalette.HighlightedText, QColor("#081712"))
    application.setPalette(palette)
    application.setStyleSheet(STYLE_SHEET)

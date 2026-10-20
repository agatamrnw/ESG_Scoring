
from pathlib import Path
import re
import sys

import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import QProcess, Qt, QUrl
from PyQt6.QtGui import QDesktopServices
from PyQt6.QtWidgets import (
    QApplication, QComboBox, QFrame, QHBoxLayout, QHeaderView,
    QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton,
    QProgressBar, QTabWidget, QTableWidget, QTableWidgetItem,
    QTextEdit, QVBoxLayout, QWidget
)

ROOT = Path(__file__).resolve().parent
SCRIPT = ROOT / "sec_financial_dataset.py"
RESULT = ROOT / "outputs" / "sec_financial" / "financial_esg_sec_2021.csv"

BACKGROUND = "#F7F6FA"
SURFACE = "#FFFFFF"
SURFACE_ALT = "#F2EFFF"
PURPLE = "#6B35FF"
PURPLE_LIGHT = "#A99BFF"
TEXT = "#252238"
MUTED = "#888399"
BORDER = "#E8E4F1"

STYLE = """
QWidget {
    background-color: #F7F6FA;
    color: #252238;
    font-family: Arial;
    font-size: 13px;
}

QMainWindow {
    background-color: #F7F6FA;
}

QFrame#card {
    background-color: #FFFFFF;
    border: 1px solid #E8E4F1;
    border-radius: 15px;
}

QFrame#card QLabel {
    background: transparent;
    border: none;
}

QLabel#heading {
    font-size: 27px;
    font-weight: bold;
    color: #252238;
}

QLabel#hint {
    color: #888399;
    font-size: 12px;
}

QLabel#number {
    font-size: 25px;
    font-weight: bold;
    color: #6B35FF;
}

QLineEdit {
    background-color: #FFFFFF;
    border: 1px solid #E2DEEB;
    border-radius: 10px;
    padding: 10px 13px;
    color: #252238;
    selection-background-color: #B7A4FF;
}

QLineEdit:focus {
    border: 1px solid #6B35FF;
}

QComboBox {
    background-color: #FFFFFF;
    border: 1px solid #E2DEEB;
    border-radius: 10px;
    padding: 10px 13px;
    color: #252238;
    min-width: 190px;
}

QComboBox:hover {
    border: 1px solid #A99BFF;
}

QComboBox QAbstractItemView {
    background-color: #FFFFFF;
    color: #252238;
    selection-background-color: #EAE5FF;
    selection-color: #6B35FF;
    border: 1px solid #E2DEEB;
}

QPushButton {
    background-color: #EAE5FF;
    color: #6131D9;
    border: none;
    border-radius: 10px;
    padding: 11px 16px;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #DED5FF;
}

QPushButton:pressed {
    background-color: #D0C2FF;
}

QPushButton:disabled {
    background-color: #F0EDF7;
    color: #ADA6BC;
}

QPushButton#primary {
    background-color: #6B35FF;
    color: #FFFFFF;
}

QPushButton#primary:hover {
    background-color: #8054FF;
}

QPushButton#primary:pressed {
    background-color: #5624DD;
}

QPushButton#primary:disabled {
    background-color: #D8CFF1;
    color: #FFFFFF;
}

QProgressBar {
    background-color: #EAE5F5;
    border: none;
    border-radius: 7px;
    height: 14px;
    text-align: center;
    color: #252238;
}

QProgressBar::chunk {
    background-color: #8054FF;
    border-radius: 7px;
}

QTabWidget::pane {
    background-color: #FFFFFF;
    border: 1px solid #E8E4F1;
    border-radius: 12px;
}

QTabBar {
    background: transparent;
}

QTabBar::tab {
    background-color: #F0EDF7;
    color: #817992;
    padding: 12px 21px;
    margin-right: 5px;
    border-top-left-radius: 10px;
    border-top-right-radius: 10px;
    font-weight: bold;
}

QTabBar::tab:selected {
    background-color: #6B35FF;
    color: #FFFFFF;
}

QTabBar::tab:hover:!selected {
    background-color: #EAE5FF;
    color: #6B35FF;
}

QTableWidget {
    background-color: #FFFFFF;
    alternate-background-color: #F9F7FF;
    gridline-color: #EEEBF5;
    border: none;
    border-radius: 9px;
    color: #252238;
    selection-background-color: #EAE5FF;
    selection-color: #5021CD;
}

QTableWidget::item {
    padding: 6px;
}

QHeaderView::section {
    background-color: #F0ECFF;
    color: #5A3AA3;
    padding: 10px;
    border: none;
    border-bottom: 1px solid #E3DAFF;
    font-weight: bold;
}

QTextEdit {
    background-color: #FFFFFF;
    color: #51496A;
    border: 1px solid #E8E4F1;
    border-radius: 10px;
    padding: 12px;
    selection-background-color: #D8CBFF;
}

QScrollBar:vertical {
    background: #F7F6FA;
    width: 10px;
    border: none;
}

QScrollBar::handle:vertical {
    background: #D4C9ED;
    border-radius: 5px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background: #A99BFF;
}

QScrollBar:horizontal {
    background: #F7F6FA;
    height: 10px;
    border: none;
}

QScrollBar::handle:horizontal {
    background: #D4C9ED;
    border-radius: 5px;
}
"""


class ESGWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("ESG Scoring | Machine learning(wkrótce)")
        self.resize(1200, 820)

        self.proc = None
        self.data = None
        self.buffer = ""

        self.build_ui()
        self.load_results(False)

    def build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        main = QVBoxLayout(central)
        main.setContentsMargins(23, 23, 23, 23)
        main.setSpacing(15)

        heading = QLabel("ESG Scoring")
        heading.setObjectName("heading")

        hint = QLabel(
            "Dane finansowe instytucji finansowych · 2021 · model ryzyka ESG"
        )
        hint.setObjectName("hint")

        main.addWidget(heading)
        main.addWidget(hint)

        metrics = QHBoxLayout()
        metrics.setSpacing(13)

        self.counters = {}

        for key, label_text in [
            ("companies", "Przedsiębiorstwa"),
            ("ok", "Pobrane z SEC"),
            ("roa", "Dostępne ROA"),
            ("roe", "Dostępne ROE")
        ]:
            card = QFrame()
            card.setObjectName("card")

            inner = QVBoxLayout(card)
            inner.setContentsMargins(18, 15, 18, 15)
            inner.setSpacing(8)

            label = QLabel(label_text)
            label.setObjectName("hint")

            number = QLabel("—")
            number.setObjectName("number")

            inner.addWidget(label)
            inner.addWidget(number)

            self.counters[key] = number
            metrics.addWidget(card)

        main.addLayout(metrics)

        control = QFrame()
        control.setObjectName("card")

        box = QVBoxLayout(control)
        box.setContentsMargins(18, 17, 18, 17)
        box.setSpacing(12)

        row = QHBoxLayout()
        row.setSpacing(10)

        self.email = QLineEdit()
        self.email.setPlaceholderText(
            "Adres e-mail kontaktowy dla SEC"
        )

        self.start = QPushButton("▶ Pobierz dane")
        self.start.setObjectName("primary")
        self.start.clicked.connect(self.run)

        self.stop = QPushButton("■ Przerwij")
        self.stop.setEnabled(False)
        self.stop.clicked.connect(self.cancel)

        reload_button = QPushButton("↻ Odśwież")
        reload_button.clicked.connect(
            lambda: self.load_results(True)
        )

        folder_button = QPushButton("Folder wyników")
        folder_button.clicked.connect(self.open_folder)

        row.addWidget(self.email, 2)
        row.addWidget(self.start)
        row.addWidget(self.stop)
        row.addWidget(reload_button)
        row.addWidget(folder_button)

        box.addLayout(row)

        self.progress = QProgressBar()
        self.progress.setValue(0)

        self.info = QLabel(
            "Gotowe. Wpisz e-mail i uruchom pobieranie "
            "lub przejrzyj istniejące wyniki."
        )
        self.info.setObjectName("hint")

        box.addWidget(self.progress)
        box.addWidget(self.info)

        main.addWidget(control)

        self.tabs = QTabWidget()
        main.addWidget(self.tabs, 1)

        table_page = QWidget()
        table_layout = QVBoxLayout(table_page)
        table_layout.setContentsMargins(15, 15, 15, 15)

        search_row = QHBoxLayout()

        self.search = QLineEdit()
        self.search.setPlaceholderText(
            "Szukaj: ticker, firma lub branża"
        )
        self.search.textChanged.connect(self.filter_table)

        self.count_label = QLabel("")
        self.count_label.setObjectName("hint")

        search_row.addWidget(self.search)
        search_row.addWidget(self.count_label)

        table_layout.addLayout(search_row)

        self.table = QTableWidget()

        self.columns = [
            ("Ticker", "ticker"),
            ("Spółka", "financial_name"),
            ("Branża", "financial_industry"),
            ("ESG Risk", "esg_risk_score"),
            ("Public ESG", "public_esg_score"),
            ("ROA %", "roa_2021_pct"),
            ("ROE %", "roe_2021_pct"),
            ("Aktywa (mld USD)", "assets_2021_usd"),
            ("Status", "status")
        ]

        self.table.setColumnCount(len(self.columns))
        self.table.setHorizontalHeaderLabels(
            [title for title, _ in self.columns]
        )

        self.table.verticalHeader().hide()
        self.table.setAlternatingRowColors(True)

        self.table.setEditTriggers(
            QTableWidget.EditTrigger.NoEditTriggers
        )

        self.table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )

        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.ResizeToContents
        )

        table_layout.addWidget(self.table)

        self.tabs.addTab(
            table_page,
            "Tabela wyników"
        )

        plot_page = QWidget()
        plot_layout = QVBoxLayout(plot_page)
        plot_layout.setContentsMargins(15, 15, 15, 15)

        self.plot_kind = QComboBox()

        self.plot_kind.addItems([
            "ROA a ESG Risk",
            "ROE a ESG Risk",
            "Aktywa a ESG Risk",
            "Rozkład ESG Risk"
        ])

        self.plot_kind.currentTextChanged.connect(
            self.draw_chart
        )

        plot_layout.addWidget(self.plot_kind)

        self.figure = Figure(
            figsize=(9, 5),
            facecolor=SURFACE
        )

        self.canvas = FigureCanvasQTAgg(self.figure)
        plot_layout.addWidget(self.canvas, 1)

        self.tabs.addTab(
            plot_page,
            "Wykresy"
        )

        log_page = QWidget()
        log_layout = QVBoxLayout(log_page)
        log_layout.setContentsMargins(15, 15, 15, 15)

        self.log = QTextEdit()
        self.log.setReadOnly(True)

        log_layout.addWidget(self.log)

        self.tabs.addTab(
            log_page,
            "Princess diaries"
        )

    def run(self):
        if (
            self.proc
            and self.proc.state()
            != QProcess.ProcessState.NotRunning
        ):
            return

        if not SCRIPT.exists():
            QMessageBox.warning(
                self,
                "Brak skryptu",
                f"Nie znaleziono:\n{SCRIPT}"
            )
            return

        email = self.email.text().strip()

        if not re.fullmatch(
            r"[^\s@]+@[^\s@]+\.[^\s@]+",
            email
        ):
            QMessageBox.warning(
                self,
                "E-mail",
                "Podaj poprawny adres e-mail kontaktowy do SEC."
            )
            return

        self.log.clear()
        self.buffer = ""
        self.progress.setValue(0)

        self.start.setEnabled(False)
        self.stop.setEnabled(True)

        self.info.setText("Uruchamianie pobierania…")

        self.proc = QProcess(self)
        self.proc.setWorkingDirectory(str(ROOT))

        self.proc.setProcessChannelMode(
            QProcess.ProcessChannelMode.MergedChannels
        )

        self.proc.readyReadStandardOutput.connect(
            self.read_output
        )

        self.proc.started.connect(
            lambda: self.proc.write(
                (email + "\n").encode()
            )
        )

        self.proc.finished.connect(self.finished)

        self.proc.errorOccurred.connect(
            lambda error: self.log.append(
                self.proc.errorString()
            )
        )

        self.proc.start(
            sys.executable,
            ["-u", str(SCRIPT)]
        )

        self.tabs.setCurrentIndex(2)

    def read_output(self):
        if not self.proc:
            return

        self.buffer += bytes(
            self.proc.readAllStandardOutput()
        ).decode(
            "utf-8",
            errors="replace"
        )

        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)

            self.log.append(line)

            match = re.search(
                r"\[(\d+)/(\d+)\]",
                line
            )

            if match:
                current, total = map(
                    int,
                    match.groups()
                )

                self.progress.setValue(
                    round((current - 1) / total * 100)
                )

                self.info.setText(
                    f"Pobieranie: {current}/{total} przedsiębiorstw"
                )

    def finished(self, code, status):
        self.read_output()

        if self.buffer.strip():
            self.log.append(self.buffer.strip())
            self.buffer = ""

        self.start.setEnabled(True)
        self.stop.setEnabled(False)

        if (
            code == 0
            and status == QProcess.ExitStatus.NormalExit
        ):
            self.progress.setValue(100)
            self.load_results(False)

            self.info.setText(
                "Pobieranie zakończone. Dane wczytane do panelu."
            )
        else:
            self.info.setText(
                f"Błąd lub przerwanie (kod: {code}). "
                "XD."
            )

        self.proc = None

    def cancel(self):
        if (
            self.proc
            and self.proc.state()
            != QProcess.ProcessState.NotRunning
        ):
            self.proc.kill()

    def load_results(self, show_message):
        if not RESULT.exists():
            if show_message:
                QMessageBox.information(
                    self,
                    "Brak danych",
                    f"Nie znaleziono pliku:\n{RESULT}"
                )

            self.draw_chart()
            return

        try:
            self.data = pd.read_csv(
                RESULT,
                low_memory=False
            )
        except Exception as error:
            QMessageBox.critical(
                self,
                "Błąd CSV",
                str(error)
            )
            return

        df = self.data

        self.counters["companies"].setText(
            str(len(df))
        )

        for key, column in [
            ("ok", "status"),
            ("roa", "roa_2021_pct"),
            ("roe", "roe_2021_pct")
        ]:
            if column not in df.columns:
                value = "—"
            elif key == "ok":
                value = str(
                    int(df[column].eq("OK").sum())
                )
            else:
                value = str(
                    int(df[column].notna().sum())
                )

            self.counters[key].setText(value)

        self.table.setRowCount(len(df))

        for index, row in df.reset_index(
            drop=True
        ).iterrows():
            for col, (_, field) in enumerate(self.columns):
                value = row.get(field)

                if pd.isna(value):
                    cell_text = "—"
                elif field == "assets_2021_usd":
                    cell_text = (
                        f"{float(value) / 1e9:,.1f}"
                    )
                elif field in {
                    "roa_2021_pct",
                    "roe_2021_pct",
                    "esg_risk_score",
                    "public_esg_score"
                }:
                    cell_text = f"{float(value):.2f}"
                else:
                    cell_text = str(value)

                item = QTableWidgetItem(cell_text)

                if col in (3, 4, 5, 6, 7):
                    item.setTextAlignment(
                        Qt.AlignmentFlag.AlignRight
                        | Qt.AlignmentFlag.AlignVCenter
                    )

                self.table.setItem(index, col, item)

        self.filter_table()
        self.draw_chart()

        self.info.setText(
            f"Wczytano {len(df)} obserwacji z {RESULT.name}."
        )

    def filter_table(self):
        term = self.search.text().strip().casefold()
        visible = 0

        for row in range(self.table.rowCount()):
            values = " ".join(
                self.table.item(row, col).text()
                for col in (0, 1, 2)
                if self.table.item(row, col)
            ).casefold()

            hidden = term not in values

            self.table.setRowHidden(
                row,
                hidden
            )

            if not hidden:
                visible += 1

        self.count_label.setText(
            f"{visible} / {self.table.rowCount()}"
        )

    def draw_chart(self):
        self.figure.clear()
        self.figure.set_facecolor(SURFACE)

        ax = self.figure.add_subplot(111)

        ax.set_facecolor(SURFACE)

        ax.tick_params(
            colors=TEXT,
            labelsize=10
        )

        ax.xaxis.label.set_color(TEXT)
        ax.yaxis.label.set_color(TEXT)
        ax.title.set_color(TEXT)

        for spine in ax.spines.values():
            spine.set_color(BORDER)

        ax.grid(
            color="#EAE5F2",
            alpha=0.9,
            linewidth=0.8
        )

        ax.set_axisbelow(True)

        if self.data is None or self.data.empty:
            ax.text(
                0.5,
                0.5,
                "Brak wyników",
                color=MUTED,
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=13
            )
        else:
            kind = self.plot_kind.currentText()

            if kind == "Rozkład ESG Risk":
                if "esg_risk_score" in self.data.columns:
                    score = pd.to_numeric(
                        self.data["esg_risk_score"],
                        errors="coerce"
                    ).dropna()

                    ax.hist(
                        score,
                        bins=12,
                        color=PURPLE_LIGHT,
                        edgecolor=SURFACE,
                        linewidth=1.5
                    )

                    ax.set_xlabel("ESG Risk Score")
                    ax.set_ylabel("Liczba spółek")
                    ax.set_title("Rozkład ocen ryzyka ESG")
            else:
                field, label = {
                    "ROA a ESG Risk": (
                        "roa_2021_pct",
                        "ROA 2021 (%)"
                    ),
                    "ROE a ESG Risk": (
                        "roe_2021_pct",
                        "ROE 2021 (%)"
                    ),
                    "Aktywa a ESG Risk": (
                        "assets_2021_usd",
                        "Aktywa 2021 (mld USD)"
                    )
                }[kind]

                if (
                    field in self.data.columns
                    and "esg_risk_score" in self.data.columns
                ):
                    points = self.data[
                        [field, "esg_risk_score"]
                    ].apply(
                        pd.to_numeric,
                        errors="coerce"
                    ).dropna()

                    if field == "assets_2021_usd":
                        x = points[field] / 1e9
                    else:
                        x = points[field]

                    ax.scatter(
                        x,
                        points["esg_risk_score"],
                        color=PURPLE,
                        edgecolors=SURFACE,
                        linewidths=0.8,
                        s=65,
                        alpha=0.82
                    )

                    ax.set_xlabel(label)
                    ax.set_ylabel("ESG Risk Score")
                    ax.set_title(
                        f"{kind} (n={len(points)})"
                    )

        self.figure.tight_layout(pad=2)
        self.canvas.draw_idle()

    def open_folder(self):
        RESULT.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        QDesktopServices.openUrl(
            QUrl.fromLocalFile(
                str(RESULT.parent)
            )
        )

    def closeEvent(self, event):
        if (
            self.proc
            and self.proc.state()
            != QProcess.ProcessState.NotRunning
        ):
            answer = QMessageBox.question(
                self,
                "Trwa pobieranie",
                "Przerwać pobieranie i zamknąć okno?"
            )

            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return

            self.proc.kill()
            self.proc.waitForFinished(2000)

        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)

    window = ESGWindow()
    window.show()

    sys.exit(app.exec())

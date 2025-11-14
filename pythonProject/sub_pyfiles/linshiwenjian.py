
import sys
import io
import matplotlib
matplotlib.use('Qt5Agg')  # Use Qt5Agg backend for matplotlib
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget,
QVBoxLayout,
                            QHBoxLayout, QLabel, QTextEdit, QPushButton,
                            QSizePolicy, QTabWidget)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

class DataAnalysisApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Data Analysis Toolkit')
        self.setGeometry(100, 100, 800, 600)

        # Main layout
        main_layout = QVBoxLayout()

        # Tabs
        self.tab_widget = QTabWidget()
        self.create_numpy_tab()
        self.create_pandas_tab()
        self.create_matplotlib_tab()

        main_layout.addWidget(self.tab_widget)
        self.setLayout(main_layout)

    def create_numpy_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        title = QLabel("NumPy")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        intro = QTextEdit()
        intro.setReadOnly(True)
        intro.setPlainText("""NumPy is a library for the Python programming
language, adding support for large, multi-dimensional arrays and matrices,
along with a large collection of high-level mathematical functions to
operate on these arrays.""")
        layout.addWidget(intro)

        code_label = QLabel("Code:")
        code_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(code_label)

        self.numpy_code = QTextEdit()
        self.numpy_code.setPlainText("""import numpy as np\n\nx = np.array([1,
2, 3])\nprint(x)\nprint(x.mean())""")
        self.numpy_code.setFont(QFont("Courier", 10))
        layout.addWidget(self.numpy_code)

        btn_layout = QHBoxLayout()
        self.numpy_btn = QPushButton("Run")
        self.numpy_btn.clicked.connect(self.execute_numpy)
        btn_layout.addWidget(self.numpy_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self.numpy_result = QTextEdit()
        self.numpy_result.setReadOnly(True)
        self.numpy_result.setFont(QFont("Courier", 10))
        layout.addWidget(self.numpy_result)

        tab.setLayout(layout)
        self.tab_widget.addTab(tab, "NumPy")

    def create_pandas_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        title = QLabel("Pandas")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        intro = QTextEdit()
        intro.setReadOnly(True)
        intro.setPlainText("""Pandas is a library providing high-performance,
easy-to-use data structures and data analysis tools for the Python
programming language.""")
        layout.addWidget(intro)

        code_label = QLabel("Code:")
        code_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(code_label)

        self.pandas_code = QTextEdit()
        self.pandas_code.setPlainText("""import pandas as pd\n\ndata =
{'Name': ['John', 'Anna', 'Peter'], 'Age': [28, 24,
35]}\nprint(pd.DataFrame(data))""")
        self.pandas_code.setFont(QFont("Courier", 10))
        layout.addWidget(self.pandas_code)

        btn_layout = QHBoxLayout()
        self.pandas_btn = QPushButton("Run")
        self.pandas_btn.clicked.connect(self.execute_pandas)
        btn_layout.addWidget(self.pandas_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        self.pandas_result = QTextEdit()
        self.pandas_result.setReadOnly(True)
        self.pandas_result.setFont(QFont("Courier", 10))
        layout.addWidget(self.pandas_result)

        tab.setLayout(layout)
        self.tab_widget.addTab(tab, "Pandas")

    def create_matplotlib_tab(self):
        tab = QWidget()
        layout = QVBoxLayout()

        title = QLabel("Matplotlib")
        title.setFont(QFont("Arial", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)

        intro = QTextEdit()
        intro.setReadOnly(True)
        intro.setPlainText("""Matplotlib is a plotting library for the Python
programming language and its numerical mathematics extension NumPy.""")
        layout.addWidget(intro)

        code_label = QLabel("Code:")
        code_label.setFont(QFont("Arial", 12, QFont.Bold))
        layout.addWidget(code_label)

        self.matplotlib_code = QTextEdit()
        self.matplotlib_code.setPlainText("""import matplotlib.pyplot as
plt\nimport numpy as np\n\nx = np.linspace(0, 10, 100)\ny =
np.sin(x)\nplt.plot(x, y)\nplt.title('Sine
Wave')\nplt.xlabel('X')\nplt.ylabel('Y')""")
        self.matplotlib_code.setFont(QFont("Courier", 10))
        layout.addWidget(self.matplotlib_code)

        btn_layout = QHBoxLayout()
        self.matplotlib_btn = QPushButton("Run")
        self.matplotlib_btn.clicked.connect(self.execute_matplotlib)
        btn_layout.addWidget(self.matplotlib_btn)
        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        # Matplotlib figure
        self.figure = Figure()
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding,
QSizePolicy.Expanding)
        layout.addWidget(self.canvas)

        self.matplotlib_result = QTextEdit()
        self.matplotlib_result.setReadOnly(True)
        self.matplotlib_result.setFont(QFont("Courier", 10))
        layout.addWidget(self.matplotlib_result)

        tab.setLayout(layout)
        self.tab_widget.addTab(tab, "Matplotlib")

    def execute_numpy(self):
        code = self.numpy_code.toPlainText()
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            exec(code)
        except Exception as e:
            self.numpy_result.setPlainText(f"Error: {str(e)}")
        finally:
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
            self.numpy_result.setPlainText(output)

    def execute_pandas(self):
        code = self.pandas_code.toPlainText()
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            exec(code)
        except Exception as e:
            self.pandas_result.setPlainText(f"Error: {str(e)}")
        finally:
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
            self.pandas_result.setPlainText(output)

    def execute_matplotlib(self):
        code = self.matplotlib_code.toPlainText()
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            exec(code)
        except Exception as e:
            self.matplotlib_result.setPlainText(f"Error: {str(e)}")
        finally:
            output = sys.stdout.getvalue()
            sys.stdout = old_stdout
            self.matplotlib_result.setPlainText(output)
            # Update the plot
            self.figure.clear()
            # We assume the code already did the plotting
            self.canvas.draw()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DataAnalysisApp()
    window.show()
    sys.exit(app.exec_())
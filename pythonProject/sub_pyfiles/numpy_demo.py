import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('qtagg')  # 新版本统一使用这个名称，自动适配Qt5/Qt6
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pyperclip
import inspect
from PyQt6.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout,
                             QHBoxLayout, QTextEdit, QLabel, QPushButton, QGroupBox,
                             QSplitter, QTreeWidget, QTreeWidgetItem, QToolTip, QMessageBox,
                             QScrollArea, QFrame, QComboBox)
from PyQt6.QtCore import Qt, QPoint, pyqtSignal, QThread
from PyQt6.QtGui import QFont, QTextCursor, QIcon, QColor, QTextCharFormat, QSyntaxHighlighter, QTextDocument

# 设置matplotlib中文字体
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号


class CodeExecutor(QThread):
    """代码执行线程，避免UI卡顿"""
    output_signal = pyqtSignal(str)
    result_signal = pyqtSignal(object)
    error_signal = pyqtSignal(str)
    finished_signal = pyqtSignal()

    def __init__(self, code, local_vars=None):
        super().__init__()
        self.code = code
        self.local_vars = local_vars if local_vars is not None else {}

        # 导入常用库到执行环境
        self.local_vars.update({
            'np': np,
            'pd': pd,
            'plt': plt,
            'Figure': Figure
        })

    def run(self):
        try:
            # 重定向print输出
            import io
            old_stdout = sys.stdout
            redirected_output = sys.stdout = io.StringIO()

            # 执行代码
            exec(self.code, globals(), self.local_vars)

            # 恢复stdout
            sys.stdout = old_stdout
            output = redirected_output.getvalue()

            # 发送输出和结果
            self.output_signal.emit(output)
            self.result_signal.emit(self.local_vars)

        except Exception as e:
            self.error_signal.emit(str(e))
        finally:
            self.finished_signal.emit()


class PythonHighlighter(QSyntaxHighlighter):
    """Python语法高亮"""

    def __init__(self, parent: QTextDocument) -> None:
        super().__init__(parent)

        # 关键字格式
        keyword_format = QTextCharFormat()
        keyword_format.setForeground(QColor("#0000FF"))
        keyword_format.setFontWeight(QFont.Weight.Bold)

        # 关键字列表
        keywords = [
            'and', 'as', 'assert', 'break', 'class', 'continue', 'def',
            'del', 'elif', 'else', 'except', 'finally', 'for', 'from',
            'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal',
            'not', 'or', 'pass', 'raise', 'return', 'try', 'while',
            'with', 'yield', 'True', 'False', 'None'
        ]

        # 函数格式
        function_format = QTextCharFormat()
        function_format.setForeground(QColor("#007020"))

        # 字符串格式
        string_format = QTextCharFormat()
        string_format.setForeground(QColor("#BA2121"))

        # 注释格式
        comment_format = QTextCharFormat()
        comment_format.setForeground(QColor("#606060"))
        comment_format.setFontItalic(True)

        # 数字格式
        number_format = QTextCharFormat()
        number_format.setForeground(QColor("#800000"))

        # 规则列表
        self.highlighting_rules = [
            (r'\b(' + '|'.join(keywords) + r')\b', keyword_format),
            (r'\bdef\s+(\w+)\s*\(', 1, function_format),
            (r'\bclass\s+(\w+)\b', 1, function_format),
            (r'"[^"]*"', string_format),
            (r"'[^']*'", string_format),
            (r'#.*$', comment_format),
            (r'\b\d+\b', number_format),
            (r'\b\d+\.\d+\b', number_format),
            (r'\b0x[0-9a-fA-F]+\b', number_format)
        ]

    def highlightBlock(self, text: str) -> None:
        for pattern, format in self.highlighting_rules[:1]:  # 关键字规则
            self._highlight_pattern(pattern, format, text)

        for pattern, group, format in self.highlighting_rules[1:3]:  # 函数和类规则
            self._highlight_pattern_group(pattern, group, format, text)

        for pattern, format in self.highlighting_rules[3:]:  # 其他规则
            self._highlight_pattern(pattern, format, text)

    def _highlight_pattern(self, pattern, format, text):
        import re
        for match in re.finditer(pattern, text):
            start, end = match.span()
            self.setFormat(start, end - start, format)

    def _highlight_pattern_group(self, pattern, group, format, text):
        import re
        for match in re.finditer(pattern, text):
            if match.group(group):
                start, end = match.span(group)
                self.setFormat(start, end - start, format)


class MplCanvas(FigureCanvasQTAgg):  # 显式继承正确的类
    """Matplotlib画布"""

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)

        # 正确初始化父类
        super().__init__(self.fig)

        # 仅当parent是QWidget时才设置
        if parent and isinstance(parent, QWidget):
            self.setParent(parent)

        self.fig.tight_layout()

class CodeDisplayWidget(QTextEdit):
    """代码显示部件，支持点击查看详情和复制"""
    code_clicked = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(QFont("Consolas", 10))
        self.highlighter = PythonHighlighter(self.document())

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            cursor = self.cursorForPosition(event.pos())
            cursor.select(QTextCursor.SelectionType.WordUnderCursor)
            selected_text = cursor.selectedText()

            if selected_text and not selected_text.isspace():
                self.code_clicked.emit(selected_text)

        super().mousePressEvent(event)

    def contextMenuEvent(self, event):
        menu = self.createStandardContextMenu()
        copy_action = menu.addAction("复制选中内容")
        copy_action.triggered.connect(self.copy_selected)
        menu.exec(event.globalPos())

    def copy_selected(self):
        text = self.textCursor().selectedText()
        if text:
            pyperclip.copy(text)


class DocumentationWidget(QWidget):
    """文档显示部件"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.layout = QVBoxLayout(self)
        self.text_edit = QTextEdit()
        self.text_edit.setReadOnly(True)
        self.text_edit.setHtml("<h3>点击代码查看文档</h3><p>在代码区域点击函数或类名查看详细文档</p>")
        self.layout.addWidget(self.text_edit)

    def set_documentation(self, item_name):
        """设置文档内容"""
        try:
            # 尝试从numpy获取文档
            if hasattr(np, item_name):
                obj = getattr(np, item_name)
                doc = inspect.getdoc(obj) or "没有找到文档"
                self.text_edit.setHtml(f"<h3>numpy.{item_name}</h3><pre>{doc}</pre>")
                return

            # 尝试从pandas获取文档
            if hasattr(pd, item_name):
                obj = getattr(pd, item_name)
                doc = inspect.getdoc(obj) or "没有找到文档"
                self.text_edit.setHtml(f"<h3>pandas.{item_name}</h3><pre>{doc}</pre>")
                return

            # 尝试从matplotlib获取文档
            if hasattr(plt, item_name):
                obj = getattr(plt, item_name)
                doc = inspect.getdoc(obj) or "没有找到文档"
                self.text_edit.setHtml(f"<h3>matplotlib.pyplot.{item_name}</h3><pre>{doc}</pre>")
                return

            # 如果找不到，尝试更通用的查找
            modules = [np, pd, plt]
            for module in modules:
                for name in dir(module):
                    if name == item_name:
                        obj = getattr(module, name)
                        doc = inspect.getdoc(obj) or "没有找到文档"
                        mod_name = module.__name__
                        self.text_edit.setHtml(f"<h3>{mod_name}.{item_name}</h3><pre>{doc}</pre>")
                        return

            # 都找不到
            self.text_edit.setHtml(f"<h3>未找到 {item_name} 的文档</h3>")

        except Exception as e:
            self.text_edit.setHtml(f"<h3>获取文档时出错</h3><p>{str(e)}</p>")


class ExampleWidget(QWidget):
    """示例展示部件"""

    def __init__(self, title, code, description, parent=None):
        super().__init__(parent)
        self.title = title
        self.code = code
        self.description = description
        self.local_vars = {}

        self.init_ui()

    def init_ui(self):
        main_layout = QVBoxLayout(self)

        # 描述
        desc_label = QLabel(f"<h3>{self.title}</h3><p>{self.description}</p>")
        desc_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        main_layout.addWidget(desc_label)

        # 代码和结果区域分割
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 代码区域
        code_group = QGroupBox("示例代码")
        code_layout = QVBoxLayout()
        self.code_display = CodeDisplayWidget()
        self.code_display.setPlainText(self.code)
        self.code_display.code_clicked.connect(self.show_documentation)
        code_layout.addWidget(self.code_display)

        # 添加执行按钮
        run_btn = QPushButton("执行代码")
        run_btn.setToolTip("执行当前代码示例，结果将显示在右侧")
        run_btn.clicked.connect(self.run_code)
        code_layout.addWidget(run_btn)

        code_group.setLayout(code_layout)
        splitter.addWidget(code_group)

        # 结果区域
        result_group = QGroupBox("执行结果")
        result_layout = QVBoxLayout()

        # 输出和图表分割
        result_splitter = QSplitter(Qt.Orientation.Vertical)

        # 输出区域
        self.output_display = QTextEdit()
        self.output_display.setReadOnly(True)
        self.output_display.setToolTip("代码执行过程中的输出信息")
        result_splitter.addWidget(self.output_display)

        # 图表区域
        self.plot_widget = QWidget()
        self.plot_layout = QVBoxLayout(self.plot_widget)
        self.canvas = MplCanvas(self, width=5, height=4, dpi=100)
        self.plot_layout.addWidget(self.canvas)
        result_splitter.addWidget(self.plot_widget)

        # 文档区域
        self.doc_widget = DocumentationWidget()
        result_splitter.addWidget(self.doc_widget)

        result_splitter.setSizes([200, 400, 200])
        result_layout.addWidget(result_splitter)
        result_group.setLayout(result_layout)

        splitter.addWidget(result_group)
        splitter.setSizes([600, 800])

        main_layout.addWidget(splitter)

    def run_code(self):
        """执行代码"""
        self.output_display.clear()
        self.canvas.axes.clear()
        self.canvas.draw()

        # 创建并启动执行线程
        self.executor = CodeExecutor(self.code, self.local_vars.copy())
        self.executor.output_signal.connect(self.append_output)
        self.executor.result_signal.connect(self.update_vars)
        self.executor.error_signal.connect(self.show_error)
        self.executor.finished_signal.connect(self.update_plot)

        self.output_display.append("开始执行代码...\n")
        self.executor.start()

    def append_output(self, text):
        """追加输出内容"""
        self.output_display.append(text)

    def update_vars(self, vars_dict):
        """更新变量"""
        self.local_vars = vars_dict

    def show_error(self, error):
        """显示错误信息"""
        self.output_display.append(f"<span style='color: red;'>错误: {error}</span>")

    def update_plot(self):
        """更新图表"""
        self.output_display.append("代码执行完成")
        self.canvas.fig.tight_layout()
        self.canvas.draw()

    def show_documentation(self, item_name):
        """显示文档"""
        self.doc_widget.set_documentation(item_name)


class NumpyExamples(QWidget):
    """NumPy示例集合"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # 创建示例选择下拉框
        self.example_selector = QComboBox()
        self.example_selector.setToolTip("选择不同的NumPy示例")
        layout.addWidget(self.example_selector)

        # 创建滚动区域
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_area.setWidget(self.scroll_content)
        layout.addWidget(self.scroll_area)

        # 添加示例
        self.add_examples()

        # 连接选择事件
        self.example_selector.currentIndexChanged.connect(self.show_selected_example)
        self.show_selected_example(0)  # 显示第一个示例

    def add_examples(self):
        """添加NumPy示例"""
        examples = [
            {
                "title": "数组创建与基本操作",
                "description": "展示NumPy数组的各种创建方法和基本属性操作",
                "code": """
import numpy as np

# 创建数组
a = np.array([1, 2, 3, 4, 5])
b = np.zeros((3, 3))
c = np.ones((2, 4))
d = np.arange(10, 30, 5)
e = np.linspace(0, 1, 5)
f = np.random.random((2, 2))

print("数组a:", a)
print("数组a的形状:", a.shape)
print("数组a的维度:", a.ndim)
print("数组a的数据类型:", a.dtype)
print("数组a的大小:", a.size)

print("\n数组b (3x3零矩阵):\\n", b)
print("数组c (2x4一矩阵):\\n", c)
print("数组d (从10到30步长5):", d)
print("数组e (0到1之间5个均匀分布的值):", e)
print("数组f (2x2随机矩阵):\\n", f)

# 数组重塑
g = np.arange(12).reshape(3, 4)
print("\n重塑后的数组g (3x4):\\n", g)
                """
            },
            {
                "title": "数组运算",
                "description": "展示NumPy数组的各种算术运算和广播功能",
                "code": """
import numpy as np

# 创建示例数组
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])
c = np.array([[1, 2], [3, 4]])
d = np.array([[5, 6], [7, 8]])

print("数组a:", a)
print("数组b:", b)

# 基本算术运算
print("\n加法:", a + b)
print("减法:", a - b)
print("乘法:", a * b)
print("除法:", a / b)
print("平方:", a **2)

print("\n数组c:\\n", c)
print("数组d:\\n", d)

# 矩阵运算
print("\n矩阵乘法:\\n", c.dot(d))
print("矩阵转置:\\n", c.T)
print("矩阵求和:", c.sum())
print("矩阵按列求和:", c.sum(axis=0))
print("矩阵按行求和:", c.sum(axis=1))
print("矩阵最大值:", c.max())
print("矩阵最小值:", c.min())
print("矩阵平均值:", c.mean())

# 广播
e = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
f = np.array([1, 2, 3])
print("\n广播示例 (e + f):\\n", e + f)
                """
            },
            {
                "title": "数组索引与切片",
                "description": "展示NumPy数组的索引、切片和迭代方法",
                "code": """
import numpy as np

# 创建示例数组
a = np.arange(10)
print("一维数组a:", a)
print("a[2] =", a[2])
print("a[2:5] =", a[2:5])
print("a[:6:2] =", a[:6:2])  # 从开始到索引6，步长2
print("a[::-1] =", a[::-1])  # 反转数组

# 二维数组
b = np.arange(12).reshape(3, 4)
print("\n二维数组b:\\n", b)
print("b[1, 2] =", b[1, 2])  # 第二行第三列
print("b[2] =", b[2])        # 第三行
print("b[:, 1] =", b[:, 1])  # 第二列
print("b[1:3, 1:3] =\\n", b[1:3, 1:3])  # 切片

# 数组迭代
print("\n遍历b的行:")
for row in b:
    print(row)

print("\n遍历b的每个元素:")
for element in b.flat:
    print(element, end=' ')
print()

# 布尔索引
c = np.random.random(10)
print("\n随机数组c:", c)
mask = c > 0.5
print("c中大于0.5的元素:", c[mask])
                """
            },
            {
                "title": "NumPy统计函数",
                "description": "展示NumPy的各种统计分析函数",
                "code": """
import numpy as np

# 创建随机数据
np.random.seed(42)  # 设置随机种子，确保结果可重现
data = np.random.normal(0, 1, 1000)  # 1000个符合正态分布的随机数
matrix = np.random.randint(0, 10, (5, 5))  # 5x5的随机整数矩阵

print("数据基本统计:")
print("平均值:", data.mean())
print("中位数:", np.median(data))
print("标准差:", data.std())
print("方差:", data.var())
print("最小值:", data.min())
print("最大值:", data.max())
print("求和:", data.sum())
print("累积和:", data.cumsum()[:10])  # 前10个累积和

print("\n矩阵:\\n", matrix)
print("每行求和:", matrix.sum(axis=1))
print("每列求平均值:", matrix.mean(axis=0))
print("矩阵最大值所在索引:", matrix.argmax())
print("每行最大值所在索引:", matrix.argmax(axis=1))

# 排序
sorted_data = np.sort(data)
print("\n排序后的前10个数据:", sorted_data[:10])
print("排序后的后10个数据:", sorted_data[-10:])

# 唯一值
values = np.array([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])
print("\n唯一值:", np.unique(values))
print("唯一值计数:", np.bincount(values))

# 绘制直方图
plt.hist(data, bins=30, alpha=0.7)
plt.title('正态分布直方图')
plt.xlabel('值')
plt.ylabel('频数')
plt.grid(True, alpha=0.3)
                """
            },
            {
                "title": "NumPy线性代数",
                "description": "展示NumPy的线性代数运算功能",
                "code": """
import numpy as np

# 创建矩阵
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("矩阵A:\\n", A)
print("矩阵B:\\n", B)

# 矩阵乘法
print("\nA × B =\\n", np.dot(A, B))

# 矩阵转置
print("A的转置:\\n", A.T)

# 行列式
print("A的行列式:", np.linalg.det(A))

# 逆矩阵
A_inv = np.linalg.inv(A)
print("A的逆矩阵:\\n", A_inv)
print("A × A_inv =\\n", np.dot(A, A_inv))  # 应该接近单位矩阵

# 解线性方程组 Ax = b
b = np.array([[5], [11]])
x = np.linalg.solve(A, b)
print("\n解方程组 Ax = b: x =\\n", x)
print("验证: A × x =\\n", np.dot(A, x))  # 应该等于b

# 特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(A)
print("\n特征值:", eigenvalues)
print("特征向量:\\n", eigenvectors)

# SVD分解
U, s, V = np.linalg.svd(A)
print("\nSVD分解:")
print("U:\\n", U)
print("奇异值:", s)
print("V:\\n", V)

# 绘制矩阵
plt.figure(figsize=(10, 5))

plt.subplot(121)
plt.imshow(A, cmap='viridis')
plt.title('矩阵A的热图')
plt.colorbar()

plt.subplot(122)
plt.imshow(B, cmap='plasma')
plt.title('矩阵B的热图')
plt.colorbar()

plt.tight_layout()
                """
            }
        ]

        self.example_widgets = []
        for i, example in enumerate(examples):
            widget = ExampleWidget(
                title=example["title"],
                code=example["code"],
                description=example["description"]
            )
            self.example_widgets.append(widget)
            self.scroll_layout.addWidget(widget)
            self.example_selector.addItem(example["title"])
            widget.hide()  # 先隐藏所有示例

    def show_selected_example(self, index):
        """显示选中的示例"""
        for i, widget in enumerate(self.example_widgets):
            if i == index:
                widget.show()
            else:
                widget.hide()


class PandasExamples(QWidget):
    """Pandas示例集合"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # 创建示例选择下拉框
        self.example_selector = QComboBox()
        self.example_selector.setToolTip("选择不同的Pandas示例")
        layout.addWidget(self.example_selector)

        # 创建滚动区域
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_area.setWidget(self.scroll_content)
        layout.addWidget(self.scroll_area)

        # 添加示例
        self.add_examples()

        # 连接选择事件
        self.example_selector.currentIndexChanged.connect(self.show_selected_example)
        self.show_selected_example(0)  # 显示第一个示例

    def add_examples(self):
        """添加Pandas示例"""
        examples = [
            {
                "title": "Series和DataFrame基本操作",
                "description": "展示Pandas中Series和DataFrame的创建与基本操作",
                "code": """
import pandas as pd
import numpy as np

# 创建Series
s = pd.Series([1, 3, 5, np.nan, 6, 8])
print("Series:")
print(s)
print("Series的索引:", s.index)
print("Series的值:", s.values)
print("Series的描述统计:\\n", s.describe())

# 创建DataFrame
dates = pd.date_range('20230101', periods=6)
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=['A', 'B', 'C', 'D'])
print("\\nDataFrame:")
print(df)

# 查看DataFrame的基本信息
print("\\nDataFrame的头部数据:\\n", df.head(3))
print("DataFrame的尾部数据:\\n", df.tail(2))
print("DataFrame的索引:", df.index)
print("DataFrame的列名:", df.columns)
print("DataFrame的值:\\n", df.values)
print("DataFrame的描述统计:\\n", df.describe())
print("DataFrame的转置:\\n", df.T)

# 排序
print("\\n按列B排序:\\n", df.sort_values(by='B'))
print("按索引排序:\\n", df.sort_index(axis=1, ascending=False))
                """
            },
            {
                "title": "DataFrame索引与选择",
                "description": "展示Pandas中DataFrame的数据选择和索引方法",
                "code": """
import pandas as pd
import numpy as np

# 创建示例DataFrame
dates = pd.date_range('20230101', periods=6)
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=['A', 'B', 'C', 'D'])

print("原始DataFrame:\\n", df)

# 选择列
print("\\n选择列A:\\n", df['A'])
print("选择列A和B:\\n", df[['A', 'B']])

# 选择行
print("\\n选择前3行:\\n", df[0:3])
print("选择2023-01-02到2023-01-04的数据:\\n", df['20230102':'20230104'])

# 使用loc选择
print("\\n使用loc选择2023-01-03的所有列:\\n", df.loc['20230103'])
print("使用loc选择2023-01-02到2023-01-04的A和C列:\\n", 
      df.loc['20230102':'20230104', ['A', 'C']])

# 使用iloc选择
print("\\n使用iloc选择第3行:\\n", df.iloc[2])
print("使用iloc选择第1到3行，第0到1列:\\n", df.iloc[0:3, 0:2])
print("使用iloc选择不连续的行和列:\\n", df.iloc[[1, 3, 5], [0, 2]])

# 布尔索引
print("\\nA列大于0的行:\\n", df[df['A'] > 0])
print("所有值大于0的元素（小于0的用NaN代替）:\\n", df[df > 0])

# 使用isin进行过滤
df2 = df.copy()
df2['E'] = ['one', 'one', 'two', 'three', 'four', 'three']
print("\\n添加了E列的DataFrame:\\n", df2)
print("E列值为one或two的行:\\n", df2[df2['E'].isin(['one', 'two'])])
                """
            },
            {
                "title": "数据清洗与处理",
                "description": "展示Pandas的数据清洗、缺失值处理和数据转换功能",
                "code": """
import pandas as pd
import numpy as np

# 创建含有缺失值的DataFrame
dates = pd.date_range('20230101', periods=6)
df = pd.DataFrame(np.random.randn(6, 4), index=dates, columns=['A', 'B', 'C', 'D'])
# 手动设置一些缺失值
df.iloc[0, 1] = np.nan
df.iloc[1, 2] = np.nan
df.iloc[3, :] = np.nan

print("含有缺失值的DataFrame:\\n", df)

# 检查缺失值
print("\\n缺失值检查（True表示缺失）:\\n", df.isnull())

# 处理缺失值
print("\\n删除含有缺失值的行:\\n", df.dropna(how='any'))
print("填充缺失值（用0填充）:\\n", df.fillna(value=0))
print("填充缺失值（用均值填充）:\\n", df.fillna(df.mean()))

# 数据转换
print("\\n原始DataFrame:\\n", df)
print("A列大于0的元素取平方，否则取立方:\\n", 
      df['A'].apply(lambda x: x**2 if x > 0 else x**3))
print("每列的最大值减去最小值:\\n", df.apply(lambda x: x.max() - x.min()))

# 数据合并
df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
                    'B': ['B0', 'B1', 'B2', 'B3'],
                    'key': ['K0', 'K1', 'K2', 'K3']})

df2 = pd.DataFrame({'C': ['C0', 'C1', 'C2', 'C3'],
                    'D': ['D0', 'D1', 'D2', 'D3'],
                    'key': ['K0', 'K1', 'K2', 'K4']})

print("\\nDataFrame df1:\\n", df1)
print("DataFrame df2:\\n", df2)
print("内连接（只保留匹配的键）:\\n", pd.merge(df1, df2, on='key'))
print("外连接（保留所有键）:\\n", pd.merge(df1, df2, on='key', how='outer'))

# 数据分组
df3 = pd.DataFrame({
    'A': ['foo', 'bar', 'foo', 'bar', 'foo', 'bar', 'foo', 'foo'],
    'B': ['one', 'one', 'two', 'three', 'two', 'two', 'one', 'three'],
    'C': np.random.randn(8),
    'D': np.random.randn(8)
})

print("\\nDataFrame df3:\\n", df3)
print("按A列分组并计算平均值:\\n", df3.groupby('A').mean())
print("按A和B列分组并计算求和:\\n", df3.groupby(['A', 'B']).sum())
                """
            },
            {
                "title": "时间序列分析",
                "description": "展示Pandas的时间序列处理功能",
                "code": """
import pandas as pd
import numpy as np

# 创建时间序列
dates = pd.date_range('2023-01-01', periods=365)
ts = pd.Series(np.random.randn(365), index=dates)
print("时间序列的前5个值:\\n", ts.head())
print("时间序列的后5个值:\\n", ts.tail())

# 时间序列基本操作
print("\\n时间序列的描述统计:\\n", ts.describe())
print("2023年2月的数据:\\n", ts['2023-02'])
print("2023年3月到5月的数据量:", len(ts['2023-03':'2023-05']))

# 重采样
print("\\n按月重采样（平均值）:\\n", ts.resample('M').mean())
print("按季度重采样（总和）:\\n", ts.resample('Q').sum())

# 时间偏移
print("\\n原始时间索引的前5个:", ts.index[:5].tolist())
print("向前偏移1天的时间索引前5个:", (ts.index + pd.Timedelta(days=1))[:5].tolist())

# 创建一个更复杂的时间序列
dates = pd.date_range('2023-01-01', periods=1000, freq='H')  # 每小时一个数据点
ts_hourly = pd.Series(np.random.randn(len(dates)), index=dates)

# 计算滚动平均值
ts_rolling = ts_hourly.rolling(window=24).mean()  # 24小时滚动平均

# 绘制时间序列图
plt.figure(figsize=(12, 6))
ts_rolling.plot(label='24小时滚动平均')
plt.title('每小时数据的24小时滚动平均值')
plt.xlabel('日期')
plt.ylabel('值')
plt.legend()
plt.grid(True, alpha=0.3)

# 季节性分析
monthly_mean = ts.resample('M').mean()
plt.figure(figsize=(10, 5))
monthly_mean.plot(kind='bar')
plt.title('每月平均值')
plt.xlabel('月份')
plt.ylabel('平均值')
plt.grid(True, axis='y', alpha=0.3)
                """
            },
            {
                "title": "数据透视表与可视化",
                "description": "展示Pandas的数据透视表功能和基本可视化方法",
                "code": """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# 创建示例数据
data = {
    '日期': pd.date_range('2023-01-01', periods=12),
    '产品': ['A', 'A', 'A', 'B', 'B', 'B', 'C', 'C', 'C', 'D', 'D', 'D'],
    '地区': ['华东', '华北', '华南', '华东', '华北', '华南', 
            '华东', '华北', '华南', '华东', '华北', '华南'],
    '销售额': np.random.randint(1000, 5000, 12),
    '销量': np.random.randint(10, 100, 12)
}
df = pd.DataFrame(data)
print("原始数据:\\n", df)

# 创建数据透视表
pivot_table = pd.pivot_table(
    df, 
    values=['销售额', '销量'],
    index=['产品'],
    columns=['地区'],
    aggfunc=np.sum,
    margins=True  # 添加汇总行和列
)
print("\\n按产品和地区汇总的销售额和销量透视表:\\n", pivot_table)

# 另一个透视表示例 - 计算平均值
pivot_mean = pd.pivot_table(
    df, 
    values='销售额',
    index=['地区'],
    columns=['产品'],
    aggfunc=np.mean
)
print("\\n各地区各产品的平均销售额:\\n", pivot_mean)

# 数据可视化
plt.figure(figsize=(12, 10))

# 1. 柱状图 - 各产品在不同地区的销售额
plt.subplot(2, 2, 1)
pivot_sales = pd.pivot_table(df, values='销售额', index='产品', columns='地区', aggfunc=np.sum)
pivot_sales.plot(kind='bar', ax=plt.gca())
plt.title('各产品在不同地区的销售额')
plt.ylabel('销售额')
plt.grid(True, axis='y', alpha=0.3)

# 2. 折线图 - 销售额趋势
plt.subplot(2, 2, 2)
df.pivot(index='日期', columns='产品', values='销售额').plot(ax=plt.gca())
plt.title('各产品销售额趋势')
plt.ylabel('销售额')
plt.grid(True, alpha=0.3)

# 3. 饼图 - 各地区总销售额占比
plt.subplot(2, 2, 3)
region_sales = df.groupby('地区')['销售额'].sum()
region_sales.plot(kind='pie', autopct='%1.1f%%', ax=plt.gca())
plt.title('各地区销售额占比')
plt.ylabel('')  # 去除y轴标签

# 4. 散点图 - 销量与销售额的关系
plt.subplot(2, 2, 4)
for product in df['产品'].unique():
    subset = df[df['产品'] == product]
    plt.scatter(subset['销量'], subset['销售额'], label=product, alpha=0.7)
plt.title('销量与销售额的关系')
plt.xlabel('销量')
plt.ylabel('销售额')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
                """
            }
        ]

        self.example_widgets = []
        for i, example in enumerate(examples):
            widget = ExampleWidget(
                title=example["title"],
                code=example["code"],
                description=example["description"]
            )
            self.example_widgets.append(widget)
            self.scroll_layout.addWidget(widget)
            self.example_selector.addItem(example["title"])
            widget.hide()  # 先隐藏所有示例

    def show_selected_example(self, index):
        """显示选中的示例"""
        for i, widget in enumerate(self.example_widgets):
            if i == index:
                widget.show()
            else:
                widget.hide()


class MatplotlibExamples(QWidget):
    """Matplotlib示例集合"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout(self)

        # 创建示例选择下拉框
        self.example_selector = QComboBox()
        self.example_selector.setToolTip("选择不同的Matplotlib示例")
        layout.addWidget(self.example_selector)

        # 创建滚动区域
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_content = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_content)
        self.scroll_area.setWidget(self.scroll_content)
        layout.addWidget(self.scroll_area)

        # 添加示例
        self.add_examples()

        # 连接选择事件
        self.example_selector.currentIndexChanged.connect(self.show_selected_example)
        self.show_selected_example(0)  # 显示第一个示例

    def add_examples(self):
        """添加Matplotlib示例"""
        examples = [
            {
                "title": "基本图表类型",
                "description": "展示Matplotlib支持的各种基本图表类型",
                "code": """
import numpy as np
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# 创建数据
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
categories = ['A', 'B', 'C', 'D', 'E']
values = [23, 45, 12, 67, 34]
sizes = [15, 30, 45, 10]
labels = ['苹果', '香蕉', '橙子', '梨']
explode = (0, 0.1, 0, 0)

# 创建一个2x2的子图
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# 1. 折线图
axs[0, 0].plot(x, y1, label='正弦曲线', color='blue', linestyle='-', linewidth=2)
axs[0, 0].plot(x, y2, label='余弦曲线', color='red', linestyle='--', linewidth=2)
axs[0, 0].set_title('折线图示例')
axs[0, 0].set_xlabel('X轴')
axs[0, 0].set_ylabel('Y轴')
axs[0, 0].grid(True, alpha=0.3)
axs[0, 0].legend()

# 2. 柱状图
axs[0, 1].bar(categories, values, color=['red', 'green', 'blue', 'yellow', 'purple'])
axs[0, 1].set_title('柱状图示例')
axs[0, 1].set_xlabel('类别')
axs[0, 1].set_ylabel('值')
axs[0, 1].grid(True, axis='y', alpha=0.3)

# 3. 饼图
axs[1, 0].pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
              shadow=True, startangle=90)
axs[1, 0].axis('equal')  # 保证饼图是正圆形
axs[1, 0].set_title('饼图示例')

# 4. 散点图
x_scatter = np.random.randn(100)
y_scatter = np.random.randn(100)
colors = np.random.rand(100)
sizes_scatter = 100 * np.random.rand(100)

axs[1, 1].scatter(x_scatter, y_scatter, c=colors, s=sizes_scatter, alpha=0.5, cmap='viridis')
axs[1, 1].set_title('散点图示例')
axs[1, 1].set_xlabel('X轴')
axs[1, 1].set_ylabel('Y轴')
axs[1, 1].grid(True, alpha=0.3)

# 调整子图之间的间距
plt.tight_layout()
                """
            },
            {
                "title": "高级图表与自定义",
                "description": "展示Matplotlib的高级图表和自定义功能",
                "code": """
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# 创建一个2x2的子图
fig, axs = plt.subplots(2, 2, figsize=(14, 12))

# 1. 直方图和核密度估计
np.random.seed(42)
data = np.random.normal(0, 1, 1000)  # 生成1000个符合正态分布的随机数

axs[0, 0].hist(data, bins=30, density=True, alpha=0.7, color='skyblue', label='直方图')

# 添加核密度估计
from scipy import stats
kde = stats.gaussian_kde(data)
x_range = np.linspace(data.min(), data.max(), 100)
axs[0, 0].plot(x_range, kde(x_range), color='red', linewidth=2, label='核密度估计')

axs[0, 0].set_title('直方图与核密度估计')
axs[0, 0].set_xlabel('值')
axs[0, 0].set_ylabel('密度')
axs[0, 0].grid(True, alpha=0.3)
axs[0, 0].legend()

# 2. 箱线图
data1 = np.random.normal(0, 1, 100)
data2 = np.random.normal(2, 1, 100)
data3 = np.random.normal(-2, 1, 100)

axs[0, 1].boxplot([data1, data2, data3], labels=['组1', '组2', '组3'])
axs[0, 1].set_title('箱线图示例')
axs[0, 1].set_ylabel('值')
axs[0, 1].grid(True, alpha=0.3)

# 3. 热力图
# 创建一个随机矩阵
matrix = np.random.rand(10, 10)
im = axs[1, 0].imshow(matrix, cmap='viridis')

# 添加颜色条
cbar = axs[1, 0].figure.colorbar(im, ax=axs[1, 0])
cbar.ax.set_ylabel('值', rotation=-90, va="bottom")

axs[1, 0].set_title('热力图示例')
axs[1, 0].set_xticks(np.arange(10))
axs[1, 0].set_yticks(np.arange(10))

# 4. 极坐标图
r = np.arange(0, 2, 0.01)
theta = 2 * np.pi * r

axs[1, 1].plot(theta, r)
axs[1, 1].set_rmax(2)
axs[1, 1].set_rticks([0.5, 1, 1.5, 2])  # 设置半径刻度
axs[1, 1].set_rlabel_position(-22.5)  # 移动半径标签
axs[1, 1].grid(True)

axs[1, 1].set_title('极坐标图示例')

# 调整子图之间的间距
plt.tight_layout()
                """
            },
            {
                "title": "3D图表",
                "description": "展示Matplotlib的3D图表绘制功能",
                "code": """
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]

# 创建一个2x2的子图
fig = plt.figure(figsize=(14, 12))

# 1. 3D线图
ax1 = fig.add_subplot(221, projection='3d')

# 生成螺旋线数据
theta = np.linspace(-4 * np.pi, 4 * np.pi, 100)
z = np.linspace(-2, 2, 100)
r = z**2 + 1
x = r * np.sin(theta)
y = r * np.cos(theta)

ax1.plot(x, y, z, label='螺旋线')
ax1.set_title('3D线图')
ax1.set_xlabel('X轴')
ax1.set_ylabel('Y轴')
ax1.set_zlabel('Z轴')
ax1.legend()

# 2. 3D散点图
ax2 = fig.add_subplot(222, projection='3d')

# 生成随机数据
np.random.seed(42)
n = 100
x = np.random.rand(n)
y = np.random.rand(n)
z = np.random.rand(n)
colors = np.random.rand(n)
sizes = 100 * np.random.rand(n)

ax2.scatter(x, y, z, c=colors, s=sizes, alpha=0.6, cmap='viridis')
ax2.set_title('3D散点图')
ax2.set_xlabel('X轴')
ax2.set_ylabel('Y轴')
ax2.set_zlabel('Z轴')

# 3. 3D曲面图
ax3 = fig.add_subplot(223, projection='3d')

# 生成网格数据
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)  # 生成一个正弦曲面

# 绘制曲面
surf = ax3.plot_surface(X, Y, Z, cmap='coolwarm', linewidth=0, antialiased=True)

# 添加颜色条
fig.colorbar(surf, ax=ax3, shrink=0.5, aspect=5)

ax3.set_title('3D曲面图')
ax3.set_xlabel('X轴')
ax3.set_ylabel('Y轴')
ax3.set_zlabel('Z轴')

# 4. 3D柱状图
ax4 = fig.add_subplot(224, projection='3d')

# 生成数据
x = np.arange(8)
y = np.arange(8)
X, Y = np.meshgrid(x, y)
X = X.flatten()
Y = Y.flatten()

# 为每个柱子设置随机高度
Z = np.random.rand(64) * 10

# 设置柱子的宽度和深度
dx = dy = 0.5

ax4.bar3d(X, Y, np.zeros_like(Z), dx, dy, Z, shade=True, color='skyblue')
ax4.set_title('3D柱状图')
ax4.set_xlabel('X轴')
ax4.set_ylabel('Y轴')
ax4.set_zlabel('高度')
ax4.set_xticks(x)
ax4.set_yticks(y)

# 调整布局
plt.tight_layout()
                """
            },
            {
                "title": "图表动画",
                "description": "展示Matplotlib的动画功能",
                "code": """
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import sys
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# 创建图形和轴
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# 第一个动画：正弦波的传播
x = np.linspace(0, 2 * np.pi, 100)
line1, = ax1.plot(x, np.sin(x))
ax1.set_title('正弦波动画')
ax1.set_xlabel('X轴')
ax1.set_ylabel('Y轴')
ax1.set_ylim(-1.5, 1.5)
ax1.grid(True, alpha=0.3)

# 第二个动画：随机行走
x_data = []
y_data = []
line2, = ax2.plot([], [], 'b-', linewidth=1)
point2, = ax2.plot([], [], 'ro')
ax2.set_title('随机行走动画')
ax2.set_xlabel('X轴')
ax2.set_ylabel('Y轴')
ax2.set_xlim(-10, 10)
ax2.set_ylim(-10, 10)
ax2.grid(True, alpha=0.3)

# 初始化函数
def init():
    line1.set_ydata(np.sin(x))
    line2.set_data([], [])
    point2.set_data([], [])
    return line1, line2, point2

# 更新函数
def update(frame):
    # 更新正弦波
    line1.set_ydata(np.sin(x + frame * 0.1))

    # 更新随机行走
    if frame == 0:
        x_data.clear()
        y_data.clear()
        x_data.append(0)
        y_data.append(0)

    # 随机步长
    step_x = np.random.normal(0, 0.5)
    step_y = np.random.normal(0, 0.5)

    x_data.append(x_data[-1] + step_x)
    y_data.append(y_data[-1] + step_y)

    # 调整坐标轴范围
    if len(x_data) > 1:
        x_min, x_max = min(x_data), max(x_data)
        y_min, y_max = min(y_data), max(y_data)
        ax2.set_xlim(x_min - 1, x_max + 1)
        ax2.set_ylim(y_min - 1, y_max + 1)

    line2.set_data(x_data, y_data)
    point2.set_data(x_data[-1], y_data[-1])

    return line1, line2, point2

# 创建动画
ani = FuncAnimation(
    fig, update, frames=np.linspace(0, 100, 100),
    init_func=init, blit=True, interval=50
)

plt.tight_layout()
                """
            },
            {
                "title": "与Pandas结合可视化",
                "description": "展示Matplotlib与Pandas结合进行数据可视化的方法",
                "code": """
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams["axes.unicode_minus"] = False  # 正确显示负号

# 创建示例数据
np.random.seed(42)
dates = pd.date_range('2023-01-01', periods=12)

# 创建一个包含多个指标的DataFrame
data = {
    '销售额': np.random.randint(1000, 5000, 12),
    '利润': np.random.randint(100, 1000, 12),
    '客户数量': np.random.randint(50, 200, 12)
}
df = pd.DataFrame(data, index=dates)

# 创建一个2x2的子图
fig, axs = plt.subplots(2, 2, figsize=(14, 12))

# 1. 折线图 - 时间序列
df.plot(ax=axs[0, 0])
axs[0, 0].set_title('时间序列趋势图')
axs[0, 0].set_xlabel('日期')
axs[0, 0].set_ylabel('数值')
axs[0, 0].grid(True, alpha=0.3)
axs[0, 0].legend()

# 2. 柱状图 - 月度比较
monthly_data = df.resample('M').sum()
monthly_data.plot(kind='bar', ax=axs[0, 1])
axs[0, 1].set_title('月度数据汇总')
axs[0, 1].set_xlabel('月份')
axs[0, 1].set_ylabel('数值')
axs[0, 1].grid(True, axis='y', alpha=0.3)
axs[0, 1].legend()

# 3. 散点图 - 销售额与利润的关系
axs[1, 0].scatter(df['销售额'], df['利润'], color='red', alpha=0.7)
axs[1, 0].set_title('销售额与利润的关系')
axs[1, 0].set_xlabel('销售额')
axs[1, 0].set_ylabel('利润')
axs[1, 0].grid(True, alpha=0.3)

# 添加回归线
z = np.polyfit(df['销售额'], df['利润'], 1)
p = np.poly1d(z)
axs[1, 0].plot(df['销售额'], p(df['销售额']), "b--")

# 4. 饼图 - 各季度销售额占比
quarterly_sales = df['销售额'].resample('Q').sum()
quarterly_sales.index = ['Q1', 'Q2', 'Q3', 'Q4']
quarterly_sales.plot(kind='pie', autopct='%1.1f%%', ax=axs[1, 1])
axs[1, 1].set_title('各季度销售额占比')
axs[1, 1].set_ylabel('')  # 去除y轴标签

# 创建另一个DataFrame用于箱线图
categories = ['A', 'B', 'C', 'D']
products_data = {
    '产品类别': np.random.choice(categories, 100),
    '价格': np.random.normal(100, 20, 100),
    '评分': np.random.uniform(1, 5, 100)
}
products_df = pd.DataFrame(products_data)

# 添加一个新的子图
plt.figure(figsize=(10, 6))
products_df.boxplot(column='价格', by='产品类别')
plt.title('不同类别产品的价格分布')
plt.suptitle('')  # 去除pandas自动添加的标题
plt.xlabel('产品类别')
plt.ylabel('价格')
plt.grid(True, alpha=0.3)

plt.tight_layout()
                """
            }
        ]

        self.example_widgets = []
        for i, example in enumerate(examples):
            widget = ExampleWidget(
                title=example["title"],
                code=example["code"],
                description=example["description"]
            )
            self.example_widgets.append(widget)
            self.scroll_layout.addWidget(widget)
            self.example_selector.addItem(example["title"])
            widget.hide()  # 先隐藏所有示例

    def show_selected_example(self, index):
        """显示选中的示例"""
        for i, widget in enumerate(self.example_widgets):
            if i == index:
                widget.show()
            else:
                widget.hide()


class MainWindow(QMainWindow):
    """主窗口"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("NumPy, Pandas & Matplotlib 示例教程")
        self.setGeometry(100, 100, 1400, 900)

        # 创建中心部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 添加标题和说明
        header = QLabel("<h1>NumPy, Pandas 与 Matplotlib 示例教程</h1>")
        header.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(header)

        description = QLabel("""
        <p>本教程展示了Python数据科学领域中三个核心库的使用方法：</p>
        <ul>
            <li><b>NumPy</b>：用于数值计算的基础库，提供高性能的多维数组对象和数学函数</li>
            <li><b>Pandas</b>：提供高效的DataFrame数据结构和数据分析工具</li>
            <li><b>Matplotlib</b>：强大的绘图库，用于创建各种静态、动态和交互式可视化</li>
        </ul>
        <p>使用方法：选择左侧标签页，然后从下拉菜单中选择示例，点击"执行代码"按钮运行。点击代码中的函数或类名可查看详细文档。</p>
        """)
        description.setWordWrap(True)
        description.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        main_layout.addWidget(description)

        # 创建标签页
        self.tabs = QTabWidget()
        self.tabs.setToolTip("选择要查看的库示例")

        # 添加各个库的示例标签页
        self.numpy_tab = NumpyExamples()
        self.pandas_tab = PandasExamples()
        self.matplotlib_tab = MatplotlibExamples()

        self.tabs.addTab(self.numpy_tab, "NumPy")
        self.tabs.addTab(self.pandas_tab, "Pandas")
        self.tabs.addTab(self.matplotlib_tab, "Matplotlib")

        main_layout.addWidget(self.tabs)

        # 状态栏显示信息
        self.statusBar().showMessage("就绪")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
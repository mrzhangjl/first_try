import sys

from PyQt5.QtCore import Qt, QDateTime, QDate, QTime, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QLineEdit, QTextEdit,
                             QCheckBox, QRadioButton, QComboBox, QSpinBox, QDoubleSpinBox,
                             QSlider, QProgressBar, QDial, QScrollBar, QDateEdit, QTimeEdit,
                             QDateTimeEdit, QCalendarWidget, QGroupBox, QListWidget,
                             QTreeWidget, QTreeWidgetItem, QTableWidget, QTableWidgetItem,
                             QMessageBox, QFileDialog, QColorDialog, QFontDialog, QSplitter,
                             QFrame, QStatusBar, QToolTip, QAction)


class PyQt5Demo(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # 设置窗口基本属性
        self.setWindowTitle('PyQt5 控件演示')
        self.setGeometry(100, 100, 1200, 800)

        # 设置全局字体
        font = QFont()
        font.setFamily("SimHei")
        font.setPointSize(10)
        self.setFont(font)

        # 创建中心部件和主布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # 创建标题标签
        title_label = QLabel('PyQt5 控件演示中心')
        title_font = QFont("SimHei", 16, QFont.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("margin: 10px; color: #2c3e50;")
        main_layout.addWidget(title_label)

        # 创建标签页控件
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet("""
            QTabBar::tab {
                background: #ecf0f1;
                color: #2c3e50;
                padding: 10px 20px;
                border: 1px solid #bdc3c7;
                border-bottom: none;
                border-radius: 8px 8px 0 0;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #3498db;
                color: white;
            }
            QTabBar::tab:hover:!selected {
                background: #bdc3c7;
            }
            QTabWidget::pane {
                border: 1px solid #bdc3c7;
                border-radius: 0 8px 8px 8px;
                background: white;
            }
        """)
        main_layout.addWidget(self.tabs)

        # 创建各个标签页
        self.create_basic_tab()
        self.create_input_tab()
        self.create_selection_tab()
        self.create_display_tab()
        self.create_container_tab()
        self.create_dialog_tab()

        # 创建状态栏
        self.statusBar = QStatusBar()
        self.setStatusBar(self.statusBar)
        self.statusBar.showMessage('就绪')

        # 创建菜单栏
        self.create_menu_bar()

        # 创建工具栏
        self.create_tool_bar()

        # 设置全局提示样式
        QToolTip.setFont(QFont("SimHei", 10))
        self.setToolTip('PyQt5 控件演示窗口')

        # 显示窗口
        self.show()

    def create_menu_bar(self):
        menubar = self.menuBar()

        # 文件菜单
        file_menu = menubar.addMenu('文件')

        # 新建动作
        new_action = QAction('新建', self)
        new_action.setShortcut('Ctrl+N')
        new_action.setStatusTip('新建文件')
        new_action.triggered.connect(lambda: self.show_info('新建', '新建文件功能'))
        file_menu.addAction(new_action)

        # 打开动作
        open_action = QAction('打开', self)
        open_action.setShortcut('Ctrl+O')
        open_action.setStatusTip('打开文件')
        open_action.triggered.connect(lambda: self.show_info('打开', '打开文件功能'))
        file_menu.addAction(open_action)

        # 保存动作
        save_action = QAction('保存', self)
        save_action.setShortcut('Ctrl+S')
        save_action.setStatusTip('保存文件')
        save_action.triggered.connect(lambda: self.show_info('保存', '保存文件功能'))
        file_menu.addAction(save_action)

        file_menu.addSeparator()

        # 退出动作
        exit_action = QAction('退出', self)
        exit_action.setShortcut('Ctrl+Q')
        exit_action.setStatusTip('退出应用程序')
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # 编辑菜单
        edit_menu = menubar.addMenu('编辑')

        # 撤销动作
        undo_action = QAction('撤销', self)
        undo_action.setShortcut('Ctrl+Z')
        undo_action.setStatusTip('撤销上一步操作')
        undo_action.triggered.connect(lambda: self.show_info('撤销', '撤销操作功能'))
        edit_menu.addAction(undo_action)

        # 重做动作
        redo_action = QAction('重做', self)
        redo_action.setShortcut('Ctrl+Y')
        redo_action.setStatusTip('重做操作')
        redo_action.triggered.connect(lambda: self.show_info('重做', '重做操作功能'))
        edit_menu.addAction(redo_action)

        edit_menu.addSeparator()

        # 剪切动作
        cut_action = QAction('剪切', self)
        cut_action.setShortcut('Ctrl+X')
        cut_action.setStatusTip('剪切选中内容')
        cut_action.triggered.connect(lambda: self.show_info('剪切', '剪切功能'))
        edit_menu.addAction(cut_action)

        # 复制动作
        copy_action = QAction('复制', self)
        copy_action.setShortcut('Ctrl+C')
        copy_action.setStatusTip('复制选中内容')
        copy_action.triggered.connect(lambda: self.show_info('复制', '复制功能'))
        edit_menu.addAction(copy_action)

        # 粘贴动作
        paste_action = QAction('粘贴', self)
        paste_action.setShortcut('Ctrl+V')
        paste_action.setStatusTip('粘贴内容')
        paste_action.triggered.connect(lambda: self.show_info('粘贴', '粘贴功能'))
        edit_menu.addAction(paste_action)

    def create_tool_bar(self):
        toolbar = self.addToolBar('工具栏')

        # 添加按钮
        new_btn = QAction('新建', self)
        new_btn.triggered.connect(lambda: self.show_info('新建', '新建文件功能'))
        toolbar.addAction(new_btn)

        open_btn = QAction('打开', self)
        open_btn.triggered.connect(lambda: self.show_info('打开', '打开文件功能'))
        toolbar.addAction(open_btn)

        save_btn = QAction('保存', self)
        save_btn.triggered.connect(lambda: self.show_info('保存', '保存文件功能'))
        toolbar.addAction(save_btn)

        toolbar.addSeparator()

        cut_btn = QAction('剪切', self)
        cut_btn.triggered.connect(lambda: self.show_info('剪切', '剪切功能'))
        toolbar.addAction(cut_btn)

        copy_btn = QAction('复制', self)
        copy_btn.triggered.connect(lambda: self.show_info('复制', '复制功能'))
        toolbar.addAction(copy_btn)

        paste_btn = QAction('粘贴', self)
        paste_btn.triggered.connect(lambda: self.show_info('粘贴', '粘贴功能'))
        toolbar.addAction(paste_btn)

    def create_basic_tab(self):
        """创建基本控件标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 使用分割器使界面更灵活
        splitter = QSplitter(Qt.Vertical)
        layout.addWidget(splitter)

        # 第一部分：标签和按钮
        group1 = QGroupBox("标签和按钮")
        group1.setStyleSheet("QGroupBox { font-weight: bold; margin-top: 10px; }")
        group1_layout = QVBoxLayout(group1)

        # 标签
        label = QLabel('这是一个标签控件 (QLabel)')
        label.setStyleSheet("color: #34495e; padding: 5px;")
        label.setToolTip("""
        QLabel 用于显示文本或图像

        属性:
        - text: 显示的文本
        - alignment: 文本对齐方式
        - pixmap: 显示的图像
        - wordWrap: 是否自动换行

        方法:
        - setText(text): 设置显示的文本
        - setPixmap(pixmap): 设置显示的图像
        - setAlignment(alignment): 设置对齐方式
        - text(): 获取当前文本
        """)
        group1_layout.addWidget(label)

        # 按钮
        btn_layout = QHBoxLayout()

        btn1 = QPushButton('普通按钮')
        btn1.setToolTip("""
        QPushButton 标准按钮

        属性:
        - text: 按钮文本
        - icon: 按钮图标
        - enabled: 是否可用
        - checkable: 是否可勾选

        方法:
        - setText(text): 设置按钮文本
        - setIcon(icon): 设置按钮图标
        - setEnabled(enabled): 设置是否可用
        - click(): 模拟点击
        - isChecked(): 检查是否被勾选

        信号:
        - clicked(): 点击时触发
        - pressed(): 按下时触发
        - released(): 释放时触发
        """)
        btn1.clicked.connect(lambda: self.show_info('按钮点击', '普通按钮被点击了'))
        btn_layout.addWidget(btn1)

        btn2 = QPushButton('复选按钮')
        btn2.setCheckable(True)
        btn2.setToolTip("可勾选的按钮，setCheckable(True)")
        btn2.clicked.connect(
            lambda checked: self.show_info('复选按钮', f'复选按钮状态: {"选中" if checked else "未选中"}'))
        btn_layout.addWidget(btn2)

        btn3 = QPushButton('禁用按钮')
        btn3.setEnabled(False)
        btn3.setToolTip("禁用的按钮，setEnabled(False)")
        btn_layout.addWidget(btn3)

        group1_layout.addLayout(btn_layout)
        splitter.addWidget(group1)

        # 第二部分：进度条和定时器
        group2 = QGroupBox("进度条和定时器")
        group2.setStyleSheet("QGroupBox { font-weight: bold; margin-top: 10px; }")
        group2_layout = QVBoxLayout(group2)

        # 进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setToolTip("""
        QProgressBar 进度条

        属性:
        - minimum: 最小值
        - maximum: 最大值
        - value: 当前值
        - textVisible: 是否显示文本

        方法:
        - setValue(value): 设置当前值
        - setRange(min, max): 设置范围
        - reset(): 重置进度条
        - value(): 获取当前值
        """)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        group2_layout.addWidget(self.progress_bar)

        # 控制进度条的按钮
        progress_btn_layout = QHBoxLayout()

        self.start_btn = QPushButton('开始')
        self.start_btn.clicked.connect(self.start_progress)
        progress_btn_layout.addWidget(self.start_btn)

        self.stop_btn = QPushButton('停止')
        self.stop_btn.clicked.connect(self.stop_progress)
        self.stop_btn.setEnabled(False)
        progress_btn_layout.addWidget(self.stop_btn)

        self.reset_btn = QPushButton('重置')
        self.reset_btn.clicked.connect(lambda: self.progress_bar.reset())
        progress_btn_layout.addWidget(self.reset_btn)

        group2_layout.addLayout(progress_btn_layout)

        # 定时器
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_progress)
        self.progress_value = 0

        splitter.addWidget(group2)

        # 添加到标签页
        self.tabs.addTab(tab, "基本控件")

    def start_progress(self):
        self.timer.start(100)
        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)

    def stop_progress(self):
        self.timer.stop()
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

    def update_progress(self):
        self.progress_value += 1
        if self.progress_value > 100:
            self.progress_value = 0
        self.progress_bar.setValue(self.progress_value)

    def create_input_tab(self):
        """创建输入控件标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        splitter = QSplitter(Qt.Vertical)
        layout.addWidget(splitter)

        # 第一部分：文本输入
        group1 = QGroupBox("文本输入控件")
        group1.setStyleSheet("QGroupBox { font-weight: bold; margin-top: 10px; }")
        group1_layout = QVBoxLayout(group1)

        # 单行文本输入
        line_edit = QLineEdit()
        line_edit.setPlaceholderText("单行文本输入 (QLineEdit)")
        line_edit.setToolTip("""
        QLineEdit 单行文本输入框

        属性:
        - text: 输入的文本
        - placeholderText: 提示文本
        - echoMode: 输入模式（正常、密码等）
        - maxLength: 最大输入长度

        方法:
        - setText(text): 设置文本
        - text(): 获取文本
        - setEchoMode(mode): 设置输入模式
        - clear(): 清空内容

        信号:
        - textChanged(text): 文本改变时触发
        - returnPressed(): 按下回车时触发
        """)
        line_edit.textChanged.connect(lambda text: self.statusBar.showMessage(f'输入内容: {text}'))
        group1_layout.addWidget(line_edit)

        # 密码输入
        password_edit = QLineEdit()
        password_edit.setPlaceholderText("密码输入")
        password_edit.setEchoMode(QLineEdit.Password)
        password_edit.setToolTip("密码模式输入框，setEchoMode(QLineEdit.Password)")
        group1_layout.addWidget(password_edit)

        # 多行文本输入
        text_edit = QTextEdit()
        text_edit.setPlaceholderText("多行文本输入 (QTextEdit)")
        text_edit.setToolTip("""
        QTextEdit 多行文本编辑框

        属性:
        - plainText: 纯文本内容
        - html: HTML内容
        - readOnly: 是否只读

        方法:
        - setPlainText(text): 设置纯文本
        - toPlainText(): 获取纯文本
        - setHtml(html): 设置HTML内容
        - toHtml(): 获取HTML内容
        - clear(): 清空内容

        信号:
        - textChanged(): 文本改变时触发
        """)
        text_edit.setMinimumHeight(80)
        group1_layout.addWidget(text_edit)

        splitter.addWidget(group1)

        # 第二部分：数字输入
        group2 = QGroupBox("数字输入控件")
        group2.setStyleSheet("QGroupBox { font-weight: bold; margin-top: 10px; }")
        group2_layout = QVBoxLayout(group2)

        num_layout = QHBoxLayout()

        # 整数输入
        spin_box = QSpinBox()
        spin_box.setRange(0, 100)
        spin_box.setValue(50)
        spin_box.setToolTip("""
        QSpinBox 整数输入框

        属性:
        - minimum: 最小值
        - maximum: 最大值
        - value: 当前值
        - singleStep: 步长

        方法:
        - setValue(value): 设置值
        - value(): 获取值
        - setRange(min, max): 设置范围
        - setSingleStep(step): 设置步长

        信号:
        - valueChanged(value): 值改变时触发
        """)
        spin_box.valueChanged.connect(lambda value: self.statusBar.showMessage(f'整数输入: {value}'))
        num_layout.addWidget(QLabel("整数输入:"))
        num_layout.addWidget(spin_box)

        # 浮点数输入
        double_spin_box = QDoubleSpinBox()
        double_spin_box.setRange(0.0, 100.0)
        double_spin_box.setValue(50.0)
        double_spin_box.setDecimals(2)
        double_spin_box.setToolTip("""
        QDoubleSpinBox 浮点数输入框

        属性:
        - minimum: 最小值
        - maximum: 最大值
        - value: 当前值
        - decimals: 小数位数
        - singleStep: 步长

        方法:
        - setValue(value): 设置值
        - value(): 获取值
        - setDecimals(decimals): 设置小数位数
        - setRange(min, max): 设置范围

        信号:
        - valueChanged(value): 值改变时触发
        """)
        double_spin_box.valueChanged.connect(lambda value: self.statusBar.showMessage(f'浮点数输入: {value}'))
        num_layout.addWidget(QLabel("浮点数输入:"))
        num_layout.addWidget(double_spin_box)

        group2_layout.addLayout(num_layout)

        # 滑块
        slider_layout = QVBoxLayout()

        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 100)
        slider.setValue(50)
        slider.setToolTip("""
        QSlider 滑块控件

        属性:
        - orientation: 方向（水平/垂直）
        - minimum: 最小值
        - maximum: 最大值
        - value: 当前值
        - singleStep: 单步值
        - pageStep: 页步值

        方法:
        - setValue(value): 设置值
        - value(): 获取值
        - setRange(min, max): 设置范围
        - setOrientation(orientation): 设置方向

        信号:
        - valueChanged(value): 值改变时触发
        - sliderMoved(value): 滑块移动时触发
        """)
        slider.valueChanged.connect(lambda value: self.statusBar.showMessage(f'滑块值: {value}'))
        slider_layout.addWidget(QLabel("滑块:"))
        slider_layout.addWidget(slider)

        # 滚动条
        scroll_bar = QScrollBar(Qt.Horizontal)
        scroll_bar.setRange(0, 100)
        scroll_bar.setValue(50)
        scroll_bar.setToolTip("""
        QScrollBar 滚动条控件

        属性:
        - orientation: 方向（水平/垂直）
        - minimum: 最小值
        - maximum: 最大值
        - value: 当前值
        - singleStep: 单步值
        - pageStep: 页步值

        方法:
        - setValue(value): 设置值
        - value(): 获取值
        - setRange(min, max): 设置范围

        信号:
        - valueChanged(value): 值改变时触发
        - sliderMoved(value): 滑块移动时触发
        """)
        scroll_bar.valueChanged.connect(lambda value: self.statusBar.showMessage(f'滚动条值: {value}'))
        slider_layout.addWidget(QLabel("滚动条:"))
        slider_layout.addWidget(scroll_bar)

        # 拨号控件
        dial = QDial()
        dial.setRange(0, 100)
        dial.setValue(50)
        dial.setToolTip("""
        QDial 拨号控件

        属性:
        - minimum: 最小值
        - maximum: 最大值
        - value: 当前值
        - wrapping: 是否可环绕

        方法:
        - setValue(value): 设置值
        - value(): 获取值
        - setRange(min, max): 设置范围
        - setWrapping(wrapping): 设置是否可环绕

        信号:
        - valueChanged(value): 值改变时触发
        """)
        dial.valueChanged.connect(lambda value: self.statusBar.showMessage(f'拨号控件值: {value}'))
        dial_layout = QHBoxLayout()
        dial_layout.addWidget(QLabel("拨号控件:"))
        dial_layout.addWidget(dial)
        slider_layout.addLayout(dial_layout)

        group2_layout.addLayout(slider_layout)

        splitter.addWidget(group2)

        # 添加到标签页
        self.tabs.addTab(tab, "输入控件")

    def create_selection_tab(self):
        """创建选择控件标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        splitter = QSplitter(Qt.Vertical)
        layout.addWidget(splitter)

        # 第一部分：复选框和单选按钮
        group1 = QGroupBox("选择按钮")
        group1.setStyleSheet("QGroupBox { font-weight: bold; margin-top: 10px; }")
        group1_layout = QVBoxLayout(group1)

        # 复选框
        checkbox_layout = QVBoxLayout()

        checkbox1 = QCheckBox("选项 1")
        checkbox1.setToolTip("""
        QCheckBox 复选框

        属性:
        - text: 显示文本
        - checked: 是否选中
        - tristate: 是否支持三态（选中、未选中、半选中）

        方法:
        - setChecked(checked): 设置是否选中
        - isChecked(): 检查是否选中
        - setText(text): 设置显示文本

        信号:
        - toggled(checked): 状态改变时触发
        """)
        checkbox1.toggled.connect(lambda checked: self.show_checkbox_status(checkbox1.text(), checked))
        checkbox_layout.addWidget(checkbox1)

        checkbox2 = QCheckBox("选项 2")
        checkbox2.toggled.connect(lambda checked: self.show_checkbox_status(checkbox2.text(), checked))
        checkbox_layout.addWidget(checkbox2)

        checkbox3 = QCheckBox("选项 3 (三态)")
        checkbox3.setTristate(True)
        checkbox3.setToolTip("三态复选框，setTristate(True)")
        checkbox3.stateChanged.connect(lambda state: self.show_tri_checkbox_status(checkbox3.text(), state))
        checkbox_layout.addWidget(checkbox3)

        group1_layout.addLayout(checkbox_layout)

        # 单选按钮
        radiogroup = QGroupBox("单选按钮组")
        radiogroup_layout = QVBoxLayout(radiogroup)

        radio1 = QRadioButton("单选 1")
        radio1.setToolTip("""
        QRadioButton 单选按钮

        属性:
        - text: 显示文本
        - checked: 是否选中

        方法:
        - setChecked(checked): 设置是否选中
        - isChecked(): 检查是否选中
        - setText(text): 设置显示文本

        信号:
        - toggled(checked): 状态改变时触发
        """)
        radio1.toggled.connect(lambda checked: self.show_radio_status(radio1.text(), checked))
        radiogroup_layout.addWidget(radio1)

        radio2 = QRadioButton("单选 2")
        radio2.toggled.connect(lambda checked: self.show_radio_status(radio2.text(), checked))
        radiogroup_layout.addWidget(radio2)

        radio3 = QRadioButton("单选 3")
        radio3.toggled.connect(lambda checked: self.show_radio_status(radio3.text(), checked))
        radiogroup_layout.addWidget(radio3)

        group1_layout.addWidget(radiogroup)

        splitter.addWidget(group1)

        # 第二部分：下拉框和日期时间
        group2 = QGroupBox("下拉选择和日期时间")
        group2.setStyleSheet("QGroupBox { font-weight: bold; margin-top: 10px; }")
        group2_layout = QVBoxLayout(group2)

        # 下拉框
        combo_layout = QHBoxLayout()

        combo = QComboBox()
        combo.addItems(["选项 1", "选项 2", "选项 3", "选项 4"])
        combo.setToolTip("""
        QComboBox 下拉框

        属性:
        - currentText: 当前文本
        - currentIndex: 当前索引
        - editable: 是否可编辑

        方法:
        - addItem(text): 添加选项
        - addItems(list): 添加多个选项
        - setCurrentIndex(index): 设置当前索引
        - currentText(): 获取当前文本
        - currentIndex(): 获取当前索引

        信号:
        - currentIndexChanged(index): 索引改变时触发
        - currentTextChanged(text): 文本改变时触发
        """)
        combo.currentIndexChanged.connect(
            lambda index: self.statusBar.showMessage(f'下拉框选择: {combo.currentText()} (索引: {index})'))
        combo_layout.addWidget(QLabel("下拉框选择:"))
        combo_layout.addWidget(combo)

        group2_layout.addLayout(combo_layout)

        # 日期时间选择
        date_layout = QVBoxLayout()

        # 日期选择
        date_edit = QDateEdit(QDate.currentDate())
        date_edit.setDisplayFormat("yyyy-MM-dd")
        date_edit.setToolTip("""
        QDateEdit 日期编辑框

        属性:
        - date: 当前日期
        - displayFormat: 显示格式

        方法:
        - setDate(date): 设置日期
        - date(): 获取日期
        - setDisplayFormat(format): 设置显示格式

        信号:
        - dateChanged(date): 日期改变时触发
        """)
        date_layout.addWidget(QLabel("日期选择:"))
        date_layout.addWidget(date_edit)

        # 时间选择
        time_edit = QTimeEdit(QTime.currentTime())
        time_edit.setDisplayFormat("HH:mm:ss")
        time_edit.setToolTip("""
        QTimeEdit 时间编辑框

        属性:
        - time: 当前时间
        - displayFormat: 显示格式

        方法:
        - setTime(time): 设置时间
        - time(): 获取时间
        - setDisplayFormat(format): 设置显示格式

        信号:
        - timeChanged(time): 时间改变时触发
        """)
        date_layout.addWidget(QLabel("时间选择:"))
        date_layout.addWidget(time_edit)

        # 日期时间选择
        datetime_edit = QDateTimeEdit(QDateTime.currentDateTime())
        datetime_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        datetime_edit.setToolTip("""
        QDateTimeEdit 日期时间编辑框

        属性:
        - dateTime: 当前日期时间
        - displayFormat: 显示格式

        方法:
        - setDateTime(datetime): 设置日期时间
        - dateTime(): 获取日期时间
        - setDisplayFormat(format): 设置显示格式

        信号:
        - dateTimeChanged(datetime): 日期时间改变时触发
        """)
        date_layout.addWidget(QLabel("日期时间选择:"))
        date_layout.addWidget(datetime_edit)

        # 日历控件
        calendar = QCalendarWidget()
        calendar.setToolTip("""
        QCalendarWidget 日历控件

        属性:
        - selectedDate: 选中的日期
        - minimumDate: 最小日期
        - maximumDate: 最大日期

        方法:
        - setSelectedDate(date): 设置选中日期
        - selectedDate(): 获取选中日期
        - setDateRange(min, max): 设置日期范围

        信号:
        - selectionChanged(): 选择改变时触发
        - clicked(date): 点击日期时触发
        """)
        calendar.clicked.connect(lambda date: self.statusBar.showMessage(f'日历选择: {date.toString("yyyy-MM-dd")}'))
        date_layout.addWidget(QLabel("日历控件:"))
        date_layout.addWidget(calendar)

        group2_layout.addLayout(date_layout)

        splitter.addWidget(group2)

        # 添加到标签页
        self.tabs.addTab(tab, "选择控件")

    def show_checkbox_status(self, text, checked):
        status = "选中" if checked else "未选中"
        self.statusBar.showMessage(f'{text}: {status}')

    def show_tri_checkbox_status(self, text, state):
        status = "选中" if state == 2 else "半选中" if state == 1 else "未选中"
        self.statusBar.showMessage(f'{text}: {status}')

    def show_radio_status(self, text, checked):
        if checked:
            self.statusBar.showMessage(f'选中: {text}')

    def create_display_tab(self):
        """创建显示控件标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        splitter = QSplitter(Qt.Vertical)
        layout.addWidget(splitter)

        # 列表控件
        group1 = QGroupBox("列表和表格控件")
        group1.setStyleSheet("QGroupBox { font-weight: bold; margin-top: 10px; }")
        group1_layout = QHBoxLayout(group1)

        # 列表控件
        list_widget = QListWidget()
        list_widget.addItems(["列表项 1", "列表项 2", "列表项 3", "列表项 4", "列表项 5"])
        list_widget.setToolTip("""
        QListWidget 列表控件

        属性:
        - count: 列表项数量
        - currentRow: 当前选中项索引

        方法:
        - addItem(item): 添加列表项
        - addItems(items): 添加多个列表项
        - setCurrentRow(row): 设置当前选中项
        - takeItem(row): 移除并返回指定项
        - currentItem(): 获取当前选中项

        信号:
        - currentRowChanged(row): 当前选中项改变时触发
        - itemClicked(item): 列表项被点击时触发
        """)
        list_widget.currentRowChanged.connect(
            lambda row: self.statusBar.showMessage(f'列表选中项: {row} - {list_widget.currentItem().text()}'))
        group1_layout.addWidget(QLabel("列表控件:"))
        group1_layout.addWidget(list_widget)

        # 树控件
        tree_widget = QTreeWidget()
        tree_widget.setHeaderLabel("树控件")
        tree_widget.setToolTip("""
        QTreeWidget 树控件

        属性:
        - headerLabels: 表头标签
        - currentItem: 当前选中项

        方法:
        - addTopLevelItem(item): 添加顶层项
        - currentItem(): 获取当前选中项
        - expandAll(): 展开所有节点
        - collapseAll(): 折叠所有节点

        信号:
        - itemClicked(item, column): 项被点击时触发
        - currentItemChanged(current, previous): 当前项改变时触发
        """)

        # 添加树节点
        root1 = QTreeWidgetItem(["根节点 1"])
        root1.addChild(QTreeWidgetItem(["子节点 1-1"]))
        root1.addChild(QTreeWidgetItem(["子节点 1-2"]))

        root2 = QTreeWidgetItem(["根节点 2"])
        root2.addChild(QTreeWidgetItem(["子节点 2-1"]))
        root2.addChild(QTreeWidgetItem(["子节点 2-2"]))
        root2.addChild(QTreeWidgetItem(["子节点 2-3"]))

        tree_widget.addTopLevelItems([root1, root2])
        tree_widget.expandAll()
        tree_widget.itemClicked.connect(lambda item, col: self.statusBar.showMessage(f'树节点: {item.text(col)}'))
        group1_layout.addWidget(QLabel("树控件:"))
        group1_layout.addWidget(tree_widget)

        splitter.addWidget(group1)

        # 表格控件
        group2 = QGroupBox("表格控件")
        group2.setStyleSheet("QGroupBox { font-weight: bold; margin-top: 10px; }")
        group2_layout = QVBoxLayout(group2)

        table_widget = QTableWidget(5, 3)
        table_widget.setHorizontalHeaderLabels(["列 1", "列 2", "列 3"])
        table_widget.setToolTip("""
        QTableWidget 表格控件

        属性:
        - rowCount: 行数
        - columnCount: 列数
        - horizontalHeaderLabels: 水平表头标签

        方法:
        - setRowCount(rows): 设置行数
        - setColumnCount(cols): 设置列数
        - setItem(row, col, item): 设置单元格内容
        - item(row, col): 获取单元格内容
        - setHorizontalHeaderLabels(labels): 设置水平表头

        信号:
        - cellClicked(row, col): 单元格被点击时触发
        - itemChanged(item): 单元格内容改变时触发
        """)

        # 填充表格数据
        for row in range(5):
            for col in range(3):
                item = QTableWidgetItem(f'单元格 ({row + 1},{col + 1})')
                table_widget.setItem(row, col, item)

        table_widget.cellClicked.connect(lambda row, col: self.statusBar.showMessage(
            f'表格单元格: ({row + 1},{col + 1}) - {table_widget.item(row, col).text()}'))
        group2_layout.addWidget(table_widget)

        splitter.addWidget(group2)

        # 添加到标签页
        self.tabs.addTab(tab, "显示控件")

    def create_container_tab(self):
        """创建容器控件标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 分组框示例
        group1 = QGroupBox("分组框 (QGroupBox)")
        group1.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #bdc3c7;
                border-radius: 8px;
                margin-top: 10px;
                padding: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        group1.setToolTip("""
        QGroupBox 分组框控件

        属性:
        - title: 标题文本
        - checked: 是否可勾选
        - enabled: 是否可用

        方法:
        - setTitle(title): 设置标题
        - setChecked(checked): 设置是否可勾选
        - setEnabled(enabled): 设置是否可用

        信号:
        - toggled(checked): 勾选状态改变时触发
        """)
        group1_layout = QVBoxLayout(group1)

        # 在分组框内添加一些控件
        for i in range(3):
            group1_layout.addWidget(QCheckBox(f'分组框内选项 {i + 1}'))

        layout.addWidget(group1)

        # 分割器示例
        splitter_label = QLabel("分割器 (QSplitter) - 可拖动调整大小")
        splitter_label.setStyleSheet("margin: 10px 0; color: #34495e;")
        layout.addWidget(splitter_label)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setToolTip("""
        QSplitter 分割器控件

        属性:
        - orientation: 方向（水平/垂直）
        - handleWidth: 手柄宽度

        方法:
        - addWidget(widget): 添加控件
        - setOrientation(orientation): 设置方向
        - setHandleWidth(width): 设置手柄宽度

        信号:
        - splitterMoved(pos, index): 分割器移动时触发
        """)

        # 左侧框架
        left_frame = QFrame()
        left_frame.setFrameShape(QFrame.StyledPanel)
        left_frame.setStyleSheet("background-color: #ecf0f1; padding: 10px;")
        left_layout = QVBoxLayout(left_frame)
        left_layout.addWidget(QLabel("左侧面板"))
        left_layout.addWidget(QPushButton("左侧按钮"))

        # 中间框架
        mid_frame = QFrame()
        mid_frame.setFrameShape(QFrame.StyledPanel)
        mid_frame.setStyleSheet("background-color: #f8f9fa; padding: 10px;")
        mid_layout = QVBoxLayout(mid_frame)
        mid_layout.addWidget(QLabel("中间面板"))
        mid_layout.addWidget(QLineEdit("中间输入框"))

        # 右侧框架
        right_frame = QFrame()
        right_frame.setFrameShape(QFrame.StyledPanel)
        right_frame.setStyleSheet("background-color: #e9ecef; padding: 10px;")
        right_layout = QVBoxLayout(right_frame)
        right_layout.addWidget(QLabel("右侧面板"))
        right_layout.addWidget(QCheckBox("右侧复选框"))

        splitter.addWidget(left_frame)
        splitter.addWidget(mid_frame)
        splitter.addWidget(right_frame)

        layout.addWidget(splitter)

        # 添加伸缩项，使分割器占满空间
        layout.addStretch()

        # 添加到标签页
        self.tabs.addTab(tab, "容器控件")

    def create_dialog_tab(self):
        """创建对话框控件标签页"""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # 消息对话框
        msg_layout = QGroupBox("消息对话框 (QMessageBox)")
        msg_layout.setStyleSheet("QGroupBox { font-weight: bold; margin-top: 10px; }")
        msg_layout_layout = QHBoxLayout(msg_layout)

        info_btn = QPushButton("信息对话框")
        info_btn.setToolTip("QMessageBox.information() - 显示信息对话框")
        info_btn.clicked.connect(lambda: QMessageBox.information(self, "信息", "这是一个信息对话框"))

        warning_btn = QPushButton("警告对话框")
        warning_btn.setToolTip("QMessageBox.warning() - 显示警告对话框")
        warning_btn.clicked.connect(lambda: QMessageBox.warning(self, "警告", "这是一个警告对话框"))

        error_btn = QPushButton("错误对话框")
        error_btn.setToolTip("QMessageBox.critical() - 显示错误对话框")
        error_btn.clicked.connect(lambda: QMessageBox.critical(self, "错误", "这是一个错误对话框"))

        question_btn = QPushButton("询问对话框")
        question_btn.setToolTip("QMessageBox.question() - 显示询问对话框")
        question_btn.clicked.connect(self.show_question_dialog)

        msg_layout_layout.addWidget(info_btn)
        msg_layout_layout.addWidget(warning_btn)
        msg_layout_layout.addWidget(error_btn)
        msg_layout_layout.addWidget(question_btn)

        layout.addWidget(msg_layout)

        # 文件对话框
        file_layout = QGroupBox("文件对话框 (QFileDialog)")
        file_layout.setStyleSheet("QGroupBox { font-weight: bold; margin-top: 10px; }")
        file_layout_layout = QHBoxLayout(file_layout)

        open_file_btn = QPushButton("打开文件")
        open_file_btn.setToolTip("""
        QFileDialog.getOpenFileName() - 打开文件对话框

        参数:
        - parent: 父窗口
        - caption: 标题
        - directory: 初始目录
        - filter: 文件过滤器

        返回值:
        - 选中的文件路径和过滤器
        """)
        open_file_btn.clicked.connect(self.open_file_dialog)

        save_file_btn = QPushButton("保存文件")
        save_file_btn.setToolTip("QFileDialog.getSaveFileName() - 保存文件对话框")
        save_file_btn.clicked.connect(self.save_file_dialog)

        open_dir_btn = QPushButton("选择目录")
        open_dir_btn.setToolTip("QFileDialog.getExistingDirectory() - 选择目录对话框")
        open_dir_btn.clicked.connect(self.open_dir_dialog)

        file_layout_layout.addWidget(open_file_btn)
        file_layout_layout.addWidget(save_file_btn)
        file_layout_layout.addWidget(open_dir_btn)

        layout.addWidget(file_layout)

        # 其他对话框
        other_layout = QGroupBox("其他对话框")
        other_layout.setStyleSheet("QGroupBox { font-weight: bold; margin-top: 10px; }")
        other_layout_layout = QHBoxLayout(other_layout)

        color_btn = QPushButton("颜色选择对话框")
        color_btn.setToolTip("QColorDialog.getColor() - 颜色选择对话框")
        color_btn.clicked.connect(self.open_color_dialog)

        font_btn = QPushButton("字体选择对话框")
        font_btn.setToolTip("QFontDialog.getFont() - 字体选择对话框")
        font_btn.clicked.connect(self.open_font_dialog)

        other_layout_layout.addWidget(color_btn)
        other_layout_layout.addWidget(font_btn)

        layout.addWidget(other_layout)

        # 添加伸缩项
        layout.addStretch()

        # 添加到标签页
        self.tabs.addTab(tab, "对话框控件")

    def show_question_dialog(self):
        reply = QMessageBox.question(self, "询问", "你确定要执行此操作吗？",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.statusBar.showMessage("用户选择了: 是")
        else:
            self.statusBar.showMessage("用户选择了: 否")

    def open_file_dialog(self):
        filename, _ = QFileDialog.getOpenFileName(self, "打开文件", "", "文本文件 (*.txt);;所有文件 (*)")
        if filename:
            self.statusBar.showMessage(f"打开文件: {filename}")

    def save_file_dialog(self):
        filename, _ = QFileDialog.getSaveFileName(self, "保存文件", "", "文本文件 (*.txt);;所有文件 (*)")
        if filename:
            self.statusBar.showMessage(f"保存文件: {filename}")

    def open_dir_dialog(self):
        dirname = QFileDialog.getExistingDirectory(self, "选择目录", "")
        if dirname:
            self.statusBar.showMessage(f"选择目录: {dirname}")

    def open_color_dialog(self):
        color = QColorDialog.getColor(Qt.blue, self, "选择颜色")
        if color.isValid():
            self.statusBar.showMessage(f"选择颜色: {color.name()}")

    def open_font_dialog(self):
        font, ok = QFontDialog.getFont(QFont("SimHei", 10), self, "选择字体")
        if ok:
            self.statusBar.showMessage(f"选择字体: {font.family()} {font.pointSize()}pt")

    def show_info(self, title, message):
        """显示信息到状态栏"""
        self.statusBar.showMessage(f'{title}: {message}')


if __name__ == '__main__':
    app = QApplication(sys.argv)

    # 设置全局样式
    app.setStyle("Fusion")

    # 确保中文显示正常
    font = QFont("SimHei")
    app.setFont(font)

    demo = PyQt5Demo()
    sys.exit(app.exec_())
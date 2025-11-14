from manim import *

class HelloManim(Scene):
    def construct(self):
        title = Text("你好，我是Codinghou!", font_size=48)
        circle = Circle(radius=1.5, color=BLUE)
        square = Square(side_length=2.5, color=GREEN)

        self.play(Write(title))
        self.wait(1)

        self.play(title.animate.to_edge(UP))

        self.play(Create(VGroup(circle, square).arrange(RIGHT)))
        self.wait(2)

        self.play(circle.animate.move_to(square.get_center()))
        self.wait(2)

        self.play(Transform(circle, square))
        self.wait(2)

        self.play(FadeOut(title, circle, square))
        self.wait(1)

config.preview = True
scene = HelloManim()
scene.render()

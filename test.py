from manim import *

class GeneratedAnimation(Scene):
    def create_table(self, headers: list[str], data: list[list[str]], title_text: str, highlight_cols: list[int] = None) -> VGroup:
        """Helper function to create a titled table."""
        table_data = [headers] + data
        table = Table(
            table_data,
            include_header=True,
            line_config={"stroke_width": 2, "color": WHITE},
            h_buff=0.5,
            v_buff=0.5
        )
        table.get_header().set_color(YELLOW)

        if highlight_cols:
            for col_index in highlight_cols:
                table.get_columns()[col_index].set_color(RED)

        title = Text(title_text, font_size=min(28, 36), color=BLUE)
        title.next_to(table, UP, buff=0.3)

        return VGroup(title, table)

    def construct(self):
        # SEGMENT: 0 | 8
        title = Text("3NF & 4NF Normalization", font_size=min(48, 36), color=YELLOW)
        self.play(Write(title))
        self.wait(2)

        # SEGMENT: 8 | 7
        subtitle = Text("Reducing Redundancy & Improving Integrity", font_size=min(32, 36), color=WHITE)
        subtitle.next_to(title, DOWN, buff=0.5)
        self.play(FadeIn(subtitle))
        self.wait(4)
        self.play(FadeOut(title), FadeOut(subtitle))
        self.wait(1)

        # SEGMENT: 15 | 8
        title_3nf = Text("Third Normal Form (3NF)", font_size=min(40, 36), color=BLUE)
        title_3nf.to_edge(UP)
        rule1_3nf = Text("1. Must be in 2nd Normal Form (2NF)", font_size=min(32, 36), color=WHITE)
        rule1_3nf.next_to(title_3nf, DOWN, buff=0.5)
        self.play(Write(title_3nf))
        self.play(FadeIn(rule1_3nf, shift=DOWN))
        self.wait(4)

        # SEGMENT: 23 | 6
        rule2_3nf = Text("2. No Transitive Dependencies", font_size=min(32, 36), color=YELLOW)
        rule2_3nf.next_to(rule1_3nf, DOWN, buff=0.3)
        self.play(FadeIn(rule2_3nf, shift=DOWN))
        self.wait(4)
        self.play(FadeOut(rule1_3nf), FadeOut(rule2_3nf))

        # SEGMENT: 29 | 8
        td_title = Text("Transitive Dependency", font_size=min(32, 36), color=WHITE)
        td_title.next_to(title_3nf, DOWN, buff=0.5)
        self.play(Write(td_title))
        self.wait(6)

        # SEGMENT: 37 | 9
        pk = Text("Primary Key (A)", font_size=min(28, 36)).shift(LEFT*4)
        non_key1 = Text("Non-Key (B)", font_size=min(28, 36))
        non_key2 = Text("Non-Key (C)", font_size=min(28, 36)).shift(RIGHT*4)

        arrow1 = Arrow(pk.get_right(), non_key1.get_left(), buff=0.1, color=GREEN)
        arrow2 = Arrow(non_key1.get_right(), non_key2.get_left(), buff=0.1, color=GREEN)

        dependency_path = VGroup(pk, non_key1, non_key2, arrow1, arrow2).move_to(ORIGIN).shift(DOWN*1)
        transitive_arrow = DashedLine(pk.get_bottom(), non_key2.get_bottom(), path_arc=-1.5, color=RED)
        transitive_label = Text("Transitive Dependency", font_size=min(24, 36), color=RED).next_to(transitive_arrow, DOWN)

        self.play(FadeOut(td_title))
        self.play(FadeIn(dependency_path))
        self.wait(3)
        self.play(Create(transitive_arrow), Write(transitive_label))
        self.wait(4)
        self.play(FadeOut(dependency_path), FadeOut(transitive_arrow), FadeOut(transitive_label))

        # SEGMENT: 46 | 11
        headers_bad = ["EmpID (PK)", "EmpName", "DeptID", "DeptName"]
        data_bad = [
            ["101", "Alice", "D1", "Sales"],
            ["102", "Bob", "D2", "Marketing"],
            ["103", "Charlie", "D1", "Sales"],
        ]
        table_bad = self.create_table(headers_bad, data_bad, "Employee Table (Violates 3NF)")
        table_bad.scale(0.8).move_to(ORIGIN)
        self.play(FadeOut(title_3nf))
        self.play(FadeIn(table_bad))
        self.wait(9)

        # SEGMENT: 57 | 10
        highlight_rect_dept = SurroundingRectangle(table_bad[1].get_columns()[2], color=RED)
        highlight_rect_dname = SurroundingRectangle(table_bad[1].get_columns()[3], color=RED)
        dep_arrow = Arrow(highlight_rect_dept.get_right(), highlight_rect_dname.get_left(), buff=0.1, color=RED)
        dep_label = Text("DeptID -> DeptName", font_size=min(24, 36), color=RED).next_to(dep_arrow, UP)
        self.play(Create(highlight_rect_dept), Create(highlight_rect_dname))
        self.play(Create(dep_arrow), Write(dep_label))
        self.wait(6)

        # SEGMENT: 67 | 4
        self.wait(4)
        self.play(FadeOut(table_bad), FadeOut(highlight_rect_dept), FadeOut(highlight_rect_dname), FadeOut(dep_arrow), FadeOut(dep_label))

        # SEGMENT: 71 | 10
        headers_emp = ["EmpID (PK)", "EmpName", "DeptID (FK)"]
        data_emp = [
            ["101", "Alice", "D1"],
            ["102", "Bob", "D2"],
            ["103", "Charlie", "D1"],
        ]
        table_emp_3nf = self.create_table(headers_emp, data_emp, "Employee Table (3NF)")
        table_emp_3nf.scale(0.8).move_to(LEFT * 3)
        self.play(FadeIn(table_emp_3nf))
        self.wait(8)

        # SEGMENT: 81 | 10
        headers_dept = ["DeptID (PK)", "DeptName"]
        data_dept = [
            ["D1", "Sales"],
            ["D2", "Marketing"],
        ]
        table_dept_3nf = self.create_table(headers_dept, data_dept, "Department Table (3NF)")
        table_dept_3nf.scale(0.8).move_to(RIGHT * 3.5)
        self.play(FadeIn(table_dept_3nf))
        self.wait(3)

        fk_pk_arrow = Arrow(table_emp_3nf[1].get_columns()[-1].get_right(), table_dept_3nf[1].get_columns()[0].get_left(), buff=0.2, color=GREEN)
        self.play(Create(fk_pk_arrow))
        self.wait(5)

        # SEGMENT: 91 | 9
        self.wait(9)
        self.play(FadeOut(table_emp_3nf), FadeOut(table_dept_3nf), FadeOut(fk_pk_arrow))

        # SEGMENT: 100 | 8
        title_4nf = Text("Fourth Normal Form (4NF)", font_size=min(40, 36), color=BLUE)
        title_4nf.to_edge(UP)
        self.play(Write(title_4nf))
        self.wait(1)

        rule1_4nf = Text("1. Must be in Boyce-Codd Normal Form (BCNF)", font_size=min(32, 36), color=WHITE)
        rule1_4nf.next_to(title_4nf, DOWN, buff=0.5)
        self.play(FadeIn(rule1_4nf, shift=DOWN))
        self.wait(5)

        # SEGMENT: 108 | 9
        rule2_4nf = Text("2. No Multi-valued Dependencies", font_size=min(32, 36), color=YELLOW)
        rule2_4nf.next_to(rule1_4nf, DOWN, buff=0.3)
        self.play(FadeIn(rule2_4nf, shift=DOWN))
        self.wait(7)
        self.play(FadeOut(rule1_4nf), FadeOut(rule2_4nf))

        # SEGMENT: 117 | 10
        mvd_title = Text("Multi-valued Dependency", font_size=min(32, 36), color=WHITE).next_to(title_4nf, DOWN, buff=0.5)
        self.play(Write(mvd_title))
        self.wait(8)
        self.play(FadeOut(mvd_title))

        # SEGMENT: 127 | 12
        headers_mvd = ["EmpID (PK)", "Skill", "Hobby"]
        data_mvd = [
            ["101", "Python", "Reading"],
            ["101", "Java", "Reading"],
        ]
        table_mvd = self.create_table(headers_mvd, data_mvd, "Employee Info (Violates 4NF)")
        table_mvd.scale(0.8).move_to(ORIGIN)
        self.play(FadeIn(table_mvd))
        self.wait(10)

        # SEGMENT: 139 | 11
        skill_col = table_mvd[1].get_columns()[1]
        hobby_col = table_mvd[1].get_columns()[2]

        skill_highlight = SurroundingRectangle(skill_col, color=RED)
        hobby_highlight = SurroundingRectangle(hobby_col, color=RED)

        independent_text = Text("Skill and Hobby are independent", font_size=min(28, 36), color=RED).next_to(table_mvd, DOWN)
        self.play(Create(skill_highlight), Create(hobby_highlight))
        self.play(Write(independent_text))
        self.wait(7)

        # SEGMENT: 150 | 12
        self.play(FadeOut(table_mvd), FadeOut(skill_highlight), FadeOut(hobby_highlight), FadeOut(independent_text))
        self.wait(1)
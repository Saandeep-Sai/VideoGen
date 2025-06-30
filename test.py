from manim import *

class GeneratedAnimation(Scene):
    def construct(self):
        # --- SCENE 1: Introduction (0-9s) ---
        title = Text("3NF & 4NF Normalization", color=BLUE, font_size=min(36, 36), width=7.0).move_to(ORIGIN)
        subtitle = Text("Advanced Database Techniques", font_size=min(28, 36), width=7.0).next_to(title, DOWN, buff=0.5)
        self.play(FadeIn(title), FadeIn(subtitle))
        self.wait(9.0)

        # --- SCENE 2: Why Normalize (9-18s) ---
        self.play(FadeOut(title), FadeOut(subtitle))
        goal_title = Text("Goal of Normalization", color=BLUE, font_size=min(36, 36), width=7.0).move_to(UP * 3)
        benefit1 = Text("Reduce Redundancy", color=GREEN, font_size=min(28, 36), width=7.0, width=6.5).move_to(LEFT * 3.5)
        benefit2 = Text("Improve Data Integrity", color=GREEN, font_size=min(28, 36), width=7.0, width=6.5).move_to(RIGHT * 3.5)
        self.play(FadeIn(goal_title))
        self.play(Write(benefit1), Write(benefit2))
        self.wait(9.0)

        # --- SCENE 3: Intro to 3NF (18-28s) ---
        self.play(FadeOut(goal_title), FadeOut(benefit1), FadeOut(benefit2))
        title_3nf = Text("Third Normal Form (3NF)", color=BLUE, font_size=min(36, 36), width=7.0).move_to(UP * 3)
        self.play(Write(title_3nf))
        self.wait(10.0)

        # --- SCENE 4: 3NF Conditions (28-38s) ---
        condition1 = Text("1. Must be in 2NF.", font_size=min(28, 36), width=7.0, width=7.0).move_to(UP * 1)
        condition2 = Text("2. No transitive dependencies.", color=RED, font_size=min(28, 36), width=7.0, width=7.0).next_to(condition1, DOWN, buff=0.5)
        self.play(Write(condition1))
        self.play(Write(condition2))
        self.wait(10.0)

        # --- SCENE 5: Transitive Dependency Explained (38-50s) ---
        self.play(FadeOut(title_3nf), FadeOut(condition1), FadeOut(condition2))
        explanation_title = Text("What is a Transitive Dependency?", color=BLUE, font_size=min(36, 36), width=7.0).move_to(UP * 3)

        A = Circle(radius=0.5, color=GREEN).move_to(LEFT * 4)
        B = Circle(radius=0.5, color=WHITE).move_to(ORIGIN)
        C = Circle(radius=0.5, color=WHITE).move_to(RIGHT * 4)
        A_label = Text("A", font_size=min(28, 36), width=7.0).move_to(A.get_center())
        B_label = Text("B", font_size=min(28, 36), width=7.0).move_to(B.get_center())
        C_label = Text("C", font_size=min(28, 36), width=7.0).move_to(C.get_center())

        arrow_ab = Arrow(A.get_right(), B.get_left(), buff=0.1)
        arrow_bc = Arrow(B.get_right(), C.get_left(), buff=0.1)
        arrow_ac = Arrow(A.get_bottom(), C.get_bottom(), path_arc=-1.2, color=RED, stroke_width=7, buff=0.5)
        transitive_label = Text("Transitive Dependency", color=RED, font_size=min(28, 36), width=7.0).next_to(arrow_ac, DOWN)

        diagram = VGroup(A, B, C, A_label, B_label, C_label, arrow_ab, arrow_bc)
        self.play(Write(explanation_title))
        self.play(FadeIn(diagram))
        self.play(Create(arrow_ac), Write(transitive_label))
        self.wait(12.0)

        # --- SCENE 6: Problem with Transitive Dependency (50-62s) ---
        self.play(FadeOut(diagram), FadeOut(transitive_label))
        problem_text = Text(
            "A non-key attribute (C) should not depend on another non-key attribute (B).",
            font_size=28,
            width=7.0,
            color=RED
        ).move_to(ORIGIN)
        self.play(Write(problem_text))
        self.wait(12.0)

        # --- SCENE 7: 3NF Example Table (62-74s) ---
        self.play(FadeOut(explanation_title), FadeOut(problem_text))
        table_3nf_problem = Table(
            [["StudentID", "Name", "DeptID", "DeptName"],
             ["101", "Ann", "D1", "Science"],
             ["102", "Ben", "D2", "Arts"],
             ["103", "Cindy", "D1", "Science"]],
            include_outer_lines=True,
            line_config={"stroke_width": 2, "color": WHITE},
        ).scale(0.6).move_to(UP*0.5)
        table_3nf_problem.get_rows()[0].set_color(BLUE)
        table_3nf_problem.get_entries((2,1)).set_color(GREEN)
        table_3nf_problem.get_entries((3,1)).set_color(GREEN)
        table_3nf_problem.get_entries((4,1)).set_color(GREEN)

        dep_title = Text("Dependencies:", font_size=min(28, 36), width=7.0).move_to(DOWN*2.5)
        dep1 = Text("StudentID -> DeptID", font_size=min(28, 36), width=7.0).next_to(dep_title, RIGHT)
        dep2 = Text("DeptID -> DeptName", font_size=min(28, 36), width=7.0).next_to(dep1, RIGHT, buff=0.5)

        self.play(Write(table_3nf_problem))
        self.play(Write(dep_title), Write(dep1), Write(dep2))
        self.wait(12.0)

        # --- SCENE 8: Highlighting the Problem (74-81s) ---
        transitive_dep_visual = Text("StudentID -> DeptName", color=RED, font_size=min(28, 36), width=7.0).move_to(DOWN*3.5)
        redundancy_box = SurroundingRectangle(table_3nf_problem.get_rows()[1].get_parts_by_text("Science")[0], color=RED)
        redundancy_box2 = SurroundingRectangle(table_3nf_problem.get_rows()[3].get_parts_by_text("Science")[0], color=RED)
        self.play(Write(transitive_dep_visual))
        self.play(Create(redundancy_box), Create(redundancy_box2))
        self.wait(7.0)

        # --- SCENE 9: 3NF Solution (81-88s) ---
        all_3nf_problem = VGroup(table_3nf_problem, dep_title, dep1, dep2, transitive_dep_visual, redundancy_box, redundancy_box2)
        self.play(FadeOut(all_3nf_problem))

        solution_title = Text("Decomposition to 3NF", color=GREEN, font_size=min(36, 36), width=7.0).move_to(UP*3)
        student_table = Table([["StudentID", "Name", "DeptID"], ["101", "Ann", "D1"], ["102", "Ben", "D2"], ["103", "Cindy", "D1"]]).scale(0.5).move_to(LEFT*3)
        dept_table = Table([["DeptID", "DeptName"], ["D1", "Science"], ["D2", "Arts"]]).scale(0.5).move_to(RIGHT*3)
        student_table.get_rows()[0].set_color(BLUE)
        dept_table.get_rows()[0].set_color(BLUE)

        self.play(Write(solution_title))
        self.play(FadeIn(student_table), FadeIn(dept_table))
        self.wait(7.0)

        # --- SCENE 10: Intro to 4NF (88-98s) ---
        self.play(FadeOut(solution_title), FadeOut(student_table), FadeOut(dept_table))
        title_4nf = Text("Fourth Normal Form (4NF)", color=BLUE, font_size=min(36, 36), width=7.0).move_to(UP*3)
        self.play(Write(title_4nf))
        self.wait(10.0)

        # --- SCENE 11: 4NF Conditions (98-109s) ---
        condition4_1 = Text("1. Must be in BCNF (stricter 3NF).", font_size=min(28, 36), width=7.0, width=7.0).move_to(UP*1)
        condition4_2 = Text("2. No multi-valued dependencies.", color=RED, font_size=min(28, 36), width=7.0, width=7.0).next_to(condition4_1, DOWN, buff=0.5)
        self.play(Write(condition4_1))
        self.play(Write(condition4_2))
        self.wait(11.0)

        # --- SCENE 12: Multi-valued Dependency Explained (109-124s) ---
        self.play(FadeOut(title_4nf), FadeOut(condition4_1), FadeOut(condition4_2))
        mvd_title = Text("What is a Multi-valued Dependency?", color=BLUE, font_size=min(36, 36), width=7.0).move_to(UP * 3)

        A_mvd = Circle(radius=0.5, color=GREEN).move_to(LEFT * 4)
        B_mvd = Circle(radius=0.5, color=WHITE).move_to(UP * 1 + RIGHT * 2)
        C_mvd = Circle(radius=0.5, color=WHITE).move_to(DOWN * 1 + RIGHT * 2)
        A_mvd_label = Text("A", font_size=min(28, 36), width=7.0).move_to(A_mvd.get_center())
        B_mvd_label = Text("{B}", font_size=min(28, 36), width=7.0).move_to(B_mvd.get_center())
        C_mvd_label = Text("{C}", font_size=min(28, 36), width=7.0).move_to(C_mvd.get_center())

        arrow_ab_mvd = DoubleArrow(A_mvd.get_right(), B_mvd.get_left(), buff=0.2, color=RED)
        arrow_ac_mvd = DoubleArrow(A_mvd.get_right(), C_mvd.get_left(), buff=0.2, color=RED)

        mvd_diagram = VGroup(A_mvd, B_mvd, C_mvd, A_mvd_label, B_mvd_label, C_mvd_label, arrow_ab_mvd, arrow_ac_mvd)
        independent_text = Text("B and C are independent", font_size=min(28, 36), width=7.0, width=7.0).move_to(DOWN*3)

        self.play(Write(mvd_title))
        self.play(FadeIn(mvd_diagram))
        self.play(Write(independent_text))
        self.wait(15.0)

        # --- SCENE 13: 4NF Example Table (124-138s) ---
        self.play(FadeOut(mvd_title), FadeOut(mvd_diagram), FadeOut(independent_text))
        table_4nf_problem = Table(
            [["ProfID", "Course", "Research Area"],
             ["P1", "DB", "AI"],
             ["P1", "DB", "ML"],
             ["P1", "Sys", "AI"],
             ["P1", "Sys", "ML"]],
            include_outer_lines=True
        ).scale(0.55).move_to(UP*0.5)
        table_4nf_problem.get_rows()[0].set_color(BLUE)
        table_4nf_problem.get_entries((2,1)).set_color(GREEN)
        table_4nf_problem.get_entries((3,1)).set_color(GREEN)
        table_4nf_problem.get_entries((4,1)).set_color(GREEN)
        table_4nf_problem.get_entries((5,1)).set_color(GREEN)

        dep4_title = Text("Dependencies:", font_size=min(28, 36), width=7.0).move_to(DOWN*2.5)
        dep4_1 = Text("ProfID ->> Course", color=RED, font_size=min(28, 36), width=7.0).next_to(dep4_title, RIGHT)
        dep4_2 = Text("ProfID ->> Research Area", color=RED, font_size=min(28, 36), width=7.0).next_to(dep4_1, RIGHT, buff=0.5)

        self.play(Write(table_4nf_problem))
        self.play(Write(dep4_title), Write(dep4_1), Write(dep4_2))
        self.wait(14.0)

        # --- SCENE 14: Highlighting 4NF Problem (138-150s) ---
        redundancy_text = Text("Redundant combinations", color=RED, font_size=min(28, 36), width=7.0).move_to(DOWN*3.5)
        problem_box = SurroundingRectangle(table_4nf_problem.get_rows()[1:], color=RED)
        self.play(Create(problem_box), Write(redundancy_text))
        self.wait(12.0)

        # --- SCENE 15: 4NF Solution (150-160s) ---
        all_4nf_problem = VGroup(table_4nf_problem, dep4_title, dep4_1, dep4_2, problem_box, redundancy_text)
        self.play(FadeOut(all_4nf_problem))

        solution_title_4nf = Text("Decomposition to 4NF", color=GREEN, font_size=min(36, 36), width=7.0).move_to(UP*3)
        prof_course_table = Table([["ProfID", "Course"], ["P1", "DB"], ["P1", "Sys"]]).scale(0.5).move_to(LEFT*3)
        prof_research_table = Table([["ProfID", "Research Area"], ["P1", "AI"], ["P1", "ML"]]).scale(0.5).move_to(RIGHT*3)
        prof_course_table.get_rows()[0].set_color(BLUE)
        prof_research_table.get_rows()[0].set_color(BLUE)

        self.play(Write(solution_title_4nf))
        self.play(FadeIn(prof_course_table), FadeIn(prof_research_table))
        self.wait(10.0)

        # --- SCENE 16: Summary 3NF (160-170s) ---
        self.play(FadeOut(solution_title_4nf), FadeOut(prof_course_table), FadeOut(prof_research_table))
        summary_title = Text("Summary", color=BLUE, font_size=min(36, 36), width=7.0).move_to(UP*3)
        summary_3nf_title = Text("3NF", color=GREEN, font_size=min(32, 36), width=7.0).move_to(UP*1.5 + LEFT*3.5)
        summary_3nf_desc = Text("Removes transitive dependencies.", font_size=min(28, 36), width=7.0, width=5.0).next_to(summary_3nf_title, DOWN)
        summary_3nf_visual = Text("A -> B -> C", font_size=min(28, 36), width=7.0).next_to(summary_3nf_desc, DOWN)

        group_3nf = VGroup(summary_3nf_title, summary_3nf_desc, summary_3nf_visual)
        self.play(Write(summary_title))
        self.play(Write(group_3nf))
        self.wait(10.0)

        # --- SCENE 17: Summary 4NF & Conclusion (170-180s) ---
        summary_4nf_title = Text("4NF", color=GREEN, font_size=min(32, 36), width=7.0).move_to(UP*1.5 + RIGHT*3.5)
        summary_4nf_desc = Text("Removes multi-valued dependencies.", font_size=min(28, 36), width=7.0, width=5.0).next_to(summary_4nf_title, DOWN)
        summary_4nf_visual = Text("A ->> B | C", font_size=min(28, 36), width=7.0).next_to(summary_4nf_desc, DOWN)

        group_4nf = VGroup(summary_4nf_title, summary_4nf_desc, summary_4nf_visual)
        self.play(Write(group_4nf))

        final_text = Text("Thank You For Watching!", font_size=min(28, 36), width=7.0, color=BLUE).move_to(DOWN*3)
        self.play(Write(final_text))
        self.wait(10.0)
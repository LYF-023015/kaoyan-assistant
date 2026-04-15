#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
学习计划生成器
基于考研时间、个人情况和目标制定个性化学习计划
"""

import os
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import calendar

class StudyPlanGenerator:
    """学习计划生成器"""

    def __init__(self, exam_date: str = None, target_score: Dict = None):
        """
        初始化学习计划生成器

        Args:
            exam_date: 考研日期，格式：YYYY-MM-DD
            target_score: 目标分数，如：{"政治": 70, "英语": 65, "数学": 80, "专业课": 85}
        """
        self.exam_date = exam_date or self._get_next_exam_date()
        self.target_score = target_score or {}
        self.current_date = datetime.now()

        # 计算剩余天数
        self.days_left = (datetime.strptime(self.exam_date, "%Y-%m-%d") - self.current_date).days

        # 学科权重配置
        self.subject_weights = {
            "政治": 0.15,
            "英语": 0.20,
            "数学": 0.30,
            "自动控制原理": 0.35
        }

        # 学习阶段配置
        self.phases = {
            "基础阶段": 0.4,    # 40%的时间
            "强化阶段": 0.4,    # 40%的时间
            "冲刺阶段": 0.2     # 20%的时间
        }

    def _get_next_exam_date(self) -> str:
        """获取下一次考研日期（假设12月底考试）"""
        current_year = datetime.now().year
        exam_date = f"{current_year}-12-25"
        # 如果今年已过，则使用明年
        if datetime.now() > datetime.strptime(exam_date, "%Y-%m-%d"):
            exam_date = f"{current_year + 1}-12-25"
        return exam_date

    def calculate_hours_needed(self, base_hours: Dict = None) -> Dict:
        """
        计算各学科所需学习时间

        Args:
            base_hours: 基础学习时间（小时/天）

        Returns:
            各学科的总学习时间
        """
        if base_hours is None:
            base_hours = {"政治": 2, "英语": 3, "数学": 4, "自动控制原理": 5}

        # 根据目标分数调整时间
        adjusted_hours = {}
        total_base = sum(base_hours.values())

        for subject, base_hour in base_hours.items():
            if subject in self.target_score:
                # 假设每提高10分需要增加20%的时间
                current_score = 60  # 基础分数
                target = self.target_score[subject]
                if target > current_score:
                    adjustment = 1 + 0.02 * ((target - current_score) // 10)
                    adjusted_hours[subject] = int(base_hour * adjustment)
                else:
                    adjusted_hours[subject] = base_hour
            else:
                adjusted_hours[subject] = base_hour

        # 根据权重调整
        total_adjusted = sum(adjusted_hours.values())
        weighted_hours = {}

        for subject, hours in adjusted_hours.items():
            weight = self.subject_weights.get(subject, 1.0)
            weighted_hours[subject] = int(hours * weight * total_base / total_adjusted)

        return weighted_hours

    def generate_phase_plan(self) -> Dict:
        """
        生成分阶段学习计划

        Returns:
            分阶段学习计划
        """
        phase_plan = {}
        phase_names = list(self.phases.keys())

        for i, phase_name in enumerate(phase_names):
            phase_days = int(self.days_left * self.phases[phase_name])

            if i == len(phase_names) - 1:
                phase_days = self.days_left - sum(p * self.days_left for p in list(self.phases.values())[:-1])

            phase_plan[phase_name] = {
                "duration": phase_days,
                "start_date": (self.current_date + timedelta(days=sum(int(p * self.days_left) for p in list(self.phases.values())[:i]))).strftime("%Y-%m-%d"),
                "end_date": (self.current_date + timedelta(days=sum(int(p * self.days_left) for p in list(self.phases.values())[:i+1]))).strftime("%Y-%m-%d"),
                "focus": self._get_phase_focus(phase_name),
                "daily_hours": self._get_phase_hours(phase_name)
            }

        return phase_plan

    def _get_phase_focus(self, phase: str) -> Dict:
        """获取各阶段学习重点"""
        focus = {
            "基础阶段": {
                "政治": "理解基本概念，建立知识框架",
                "英语": "背诵核心词汇，掌握语法基础",
                "数学": "理解基本概念，掌握基本计算",
                "自动控制原理": "理解基本概念，掌握典型环节"
            },
            "强化阶段": {
                "政治": "系统复习，加强记忆，做练习题",
                "英语": "大量阅读，练习写作，提高速度",
                "数学": "大量做题，总结方法，提高技巧",
                "自动控制原理": "综合应用，做典型题，掌握方法"
            },
            "冲刺阶段": {
                "政治": "背诵重点，模拟考试，查漏补缺",
                "英语": "模拟考试，练习作文，保持手感",
                "数学": "真题模拟，总结错题，调整心态",
                "自动控制原理": "真题训练，重点复习，模拟考试"
            }
        }
        return focus.get(phase, {})

    def _get_phase_hours(self, phase: str) -> Dict:
        """获取各阶段每日学习时间"""
        base_hours = self.calculate_hours_needed()

        # 冲刺阶段增加时间
        if phase == "冲刺阶段":
            return {subject: int(hours * 1.5) for subject, hours in base_hours.items()}

        return base_hours

    def generate_weekly_plan(self, week_number: int) -> Dict:
        """
        生成周学习计划

        Args:
            week_number: 第几周

        Returns:
            周学习计划
        """
        # 确定当前所在阶段
        phase_plan = self.generate_phase_plan()
        current_phase = None

        accumulated_days = 0
        for phase_name, phase_info in phase_plan.items():
            if accumulated_days <= week_number * 7 < accumulated_days + phase_info["duration"]:
                current_phase = phase_name
                break
            accumulated_days += phase_info["duration"]

        if current_phase is None:
            current_phase = "冲刺阶段"

        # 生成周计划
        weekly_plan = {
            "week": week_number,
            "phase": current_phase,
            "daily_schedule": self._generate_daily_schedule(current_phase),
            "weekly_goals": self._generate_weekly_goals(current_phase, week_number),
            "milestones": self._generate_weekly_milestones(week_number)
        }

        return weekly_plan

    def _generate_daily_schedule(self, phase: str) -> Dict:
        """生成每日学习时间表"""
        hours = self._get_phase_hours(phase)

        # 根据阶段调整时间分配
        if phase == "基础阶段":
            # 上午精力好，安排难科目
            morning = {"数学": hours["数学"] // 2, "自动控制原理": hours["自动控制原理"] // 2}
            afternoon = {"英语": hours["英语"], "政治": hours["政治"]}
            evening = {"数学": hours["数学"] // 2, "自动控制原理": hours["自动控制原理"] // 2}
        elif phase == "强化阶段":
            # 平衡各科
            morning = {"数学": hours["math"] // 2, "自动控制原理": hours["自动控制原理"] // 2}
            afternoon = {"英语": hours["英语"], "政治": hours["政治"]}
            evening = {"数学": hours["math"] // 2, "自动控制原理": hours["自动控制原理"] // 2}
        else:  # 冲刺阶段
            # 重点是模拟和复习
            morning = {"政治": hours["政治"], "英语": hours["英语"]}
            afternoon = {"数学": hours["数学"], "自动控制原理": hours["自动控制原理"]}
            evening = {"总复习": 2, "查漏补缺": 1}

        return {
            "morning": morning,
            "afternoon": afternoon,
            "evening": evening
        }

    def _generate_weekly_goals(self, phase: str, week: int) -> Dict:
        """生成周学习目标"""
        base_goals = {
            "政治": {
                "基础阶段": f"完成第{week}章学习，掌握基本概念",
                "强化阶段": f"完成第{week}章练习，正确率达到80%",
                "冲刺阶段": f"背诵第{week}章重点，做模拟题"
            },
            "英语": {
                "基础阶段": f"背诵{week*50}个核心词汇，完成阅读练习",
                "强化阶段": f"完成{week}篇阅读，练习写作模板",
                "冲刺阶段": f"做{week}套真题，写作练习"
            },
            "数学": {
                "基础阶段": f"学习第{week}章，完成基础题",
                "强化阶段": f"完成第{week}章综合题，总结方法",
                "冲刺阶段": f"做第{week}套真题，分析错题"
            },
            "自动控制原理": {
                "基础阶段": f"学习第{week}章，理解基本概念",
                "强化阶段": f"完成第{week}章典型题，掌握解题方法",
                "冲刺阶段": f"做第{week}套真题，重点复习薄弱环节"
            }
        }

        return base_goals

    def _generate_weekly_milestones(self, week: int) -> List[str]:
        """生成周里程碑"""
        milestones = [
            f"完成第{week}周所有学习任务",
            f"进行{week}次自测",
            f"整理本周错题本",
            f"各科进度检查"
        ]
        return milestones

    def generate_monthly_plan(self, month: int) -> Dict:
        """
        生成月学习计划

        Args:
            month: 第几个月

        Returns:
            月学习计划
        """
        weeks_in_month = 4
        monthly_plan = {
            "month": month,
            "overall_goal": self._get_monthly_goal(month),
            "weekly_plans": [],
            "key_milestones": self._get_monthly_milestones(month),
            "resources": self._get_monthly_resources(month)
        }

        # 生成每周计划
        for week in range(1, weeks_in_month + 1):
            weekly_plan = self.generate_weekly_plan((month - 1) * weeks_in_month + week)
            monthly_plan["weekly_plans"].append(weekly_plan)

        return monthly_plan

    def _get_monthly_goal(self, month: int) -> str:
        """获取月度学习目标"""
        if month <= 3:
            return "打好基础，建立完整的知识体系"
        elif month <= 6:
            return "强化训练，提高解题能力和速度"
        else:
            return "冲刺复习，查漏补缺，调整状态"

    def _get_monthly_milestones(self, month: int) -> List[str]:
        """获取月度里程碑"""
        milestones = [
            f"完成月度学习目标",
            f"进行{month}次月考",
            f"各科进度达到{month*25}%",
            f"建立错题复习体系"
        ]
        return milestones

    def _get_monthly_resources(self, month: int) -> Dict:
        """获取月度学习资源"""
        resources = {
            "政治": ["考研政治大纲", "肖秀荣1000题", "历年真题"],
            "英语": ["考研英语词汇", "张剑黄皮书", "历年真题"],
            "数学": ["考研数学复习全书", "张宇1000题", "历年真题"],
            "自动控制原理": ["胡寿松控制原理", "南理工考研真题", "典型例题集"]
        }
        return resources

    def generate_study_plan(self, plan_type: str = "detailed") -> Dict:
        """
        生成完整学习计划

        Args:
            plan_type: 计划类型（simple/detailed）

        Returns:
            完整学习计划
        """
        study_plan = {
            "basic_info": {
                "exam_date": self.exam_date,
                "days_left": self.days_left,
                "current_date": self.current_date.strftime("%Y-%m-%d"),
                "target_scores": self.target_score
            },
            "phase_plan": self.generate_phase_plan(),
            "daily_hours": self.calculate_hours_needed(),
            "monthly_plans": []
        }

        # 生成月计划
        months = (self.days_left + 29) // 30  # 向上取整
        for month in range(1, months + 1):
            monthly_plan = self.generate_monthly_plan(month)
            study_plan["monthly_plans"].append(monthly_plan)

        if plan_type == "simple":
            # 简化版计划
            study_plan = {
                "basic_info": study_plan["basic_info"],
                "total_hours_needed": {k: v * self.days_left for k, v in study_plan["daily_hours"].items()},
                "phase_summary": {phase: info["duration"] for phase, info in study_plan["phase_plan"].items()}
            }

        return study_plan

    def export_plan(self, study_plan: Dict, filename: str = None) -> str:
        """
        导出学习计划

        Args:
            study_plan: 学习计划
            filename: 文件名

        Returns:
            文件路径
        """
        if filename is None:
            filename = f"study_plan_{datetime.now().strftime('%Y%m%d')}.json"

        filepath = os.path.join(os.path.dirname(__file__), "..", "study_plans", filename)

        # 确保目录存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # 保存计划
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(study_plan, f, ensure_ascii=False, indent=2)

        return filepath

    def print_plan_summary(self, study_plan: Dict):
        """打印计划摘要"""
        print("\n" + "="*50)
        print("考研学习计划摘要")
        print("="*50)

        # 基本信息
        info = study_plan["basic_info"]
        print(f"\n考试日期: {info['exam_date']}")
        print(f"剩余天数: {info['days_left']} 天")
        print(f"当前日期: {info['current_date']}")

        if info['target_scores']:
            print("\n目标分数:")
            for subject, score in info['target_scores'].items():
                print(f"  {subject}: {score} 分")

        # 阶段计划
        print("\n学习阶段安排:")
        for phase, info in study_plan["phase_plan"].items():
            print(f"  {phase}: {info['duration']} 天 ({info['start_date']} 至 {info['end_date']})")

        # 每日学习时间
        print("\n每日学习时间（小时）:")
        for subject, hours in study_plan["daily_hours"].items():
            print(f"  {subject}: {hours} 小时")

        print("\n" + "="*50)


# 测试代码
if __name__ == "__main__":
    # 创建生成器
    generator = StudyPlanGenerator(
        exam_date="2025-12-25",
        target_score={"政治": 75, "英语": 70, "数学": 85, "自动控制原理": 90}
    )

    # 生成详细计划
    study_plan = generator.generate_study_plan("detailed")

    # 打印摘要
    generator.print_plan_summary(study_plan)

    # 导出计划
    filepath = generator.export_plan(study_plan)
    print(f"\n计划已保存至: {filepath}")
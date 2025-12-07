"""
Goal and Monitoring Pattern Implementation

Based on: THE-BLUEPRINTS.md - Pattern 07: The Goal and Monitoring Pattern

Problem: Automated systems without a clear definition of success and a 
mechanism to monitor progress are "fire-and-forget" liabilities.

Solution: Architect an AI system that can autonomously pursue a high-level 
objective by continuously tracking its own progress against predefined 
success criteria and adapting its actions to ensure the goal is achieved.

This module provides goal definition, progress tracking, and success 
monitoring for T-RLINKOS TRM++ reasoning tasks.
"""

import time
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from enum import Enum


class GoalStatus(Enum):
    """Status of goal progress"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    ACHIEVED = "achieved"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass
class SuccessCriteria:
    """
    Defines success criteria for a reasoning goal.
    
    Supports multiple types of criteria:
    - Threshold-based (e.g., accuracy > 0.95)
    - Time-based (e.g., complete within 10 seconds)
    - Step-based (e.g., converge within 20 steps)
    - Custom predicates
    """
    name: str
    description: str
    check_function: Callable[[Dict[str, Any]], bool]
    weight: float = 1.0
    is_required: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def check(self, state: Dict[str, Any]) -> bool:
        """
        Check if criteria is met.
        
        Args:
            state: Current state dictionary
            
        Returns:
            True if criteria met, False otherwise
        """
        try:
            return self.check_function(state)
        except Exception as e:
            print(f"Error checking criteria '{self.name}': {e}")
            return False


@dataclass
class GoalDefinition:
    """
    Definition of a reasoning goal.
    
    Includes:
    - Success criteria
    - Constraints (time, steps, resources)
    - Priority
    """
    goal_id: str
    description: str
    success_criteria: List[SuccessCriteria]
    max_steps: Optional[int] = None
    max_time: Optional[float] = None
    min_score: Optional[float] = None
    priority: int = 1
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProgressSnapshot:
    """Snapshot of progress at a point in time"""
    timestamp: float
    step: int
    score: float
    criteria_met: Dict[str, bool]
    metrics: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProgressTracker:
    """
    Tracks progress toward goal achievement.
    
    Features:
    - Step-by-step progress monitoring
    - Metrics collection
    - Historical tracking
    - Progress visualization
    """
    
    def __init__(self, max_history: int = 100):
        """
        Initialize progress tracker.
        
        Args:
            max_history: Maximum number of snapshots to keep
        """
        self.max_history = max_history
        self.history: List[ProgressSnapshot] = []
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        
    def start(self):
        """Start tracking progress."""
        self.start_time = time.time()
        self.end_time = None
        self.history.clear()
    
    def record(
        self,
        step: int,
        score: float,
        criteria_met: Dict[str, bool],
        metrics: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Record a progress snapshot.
        
        Args:
            step: Current step number
            score: Current score/performance
            criteria_met: Dictionary of criteria names to boolean (met/not met)
            metrics: Additional metrics
            metadata: Additional metadata
        """
        snapshot = ProgressSnapshot(
            timestamp=time.time(),
            step=step,
            score=score,
            criteria_met=criteria_met,
            metrics=metrics or {},
            metadata=metadata or {}
        )
        
        self.history.append(snapshot)
        
        # Limit history size
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]
    
    def end(self):
        """End tracking progress."""
        self.end_time = time.time()
    
    def get_current_progress(self) -> Optional[ProgressSnapshot]:
        """Get most recent progress snapshot."""
        return self.history[-1] if self.history else None
    
    def get_progress_rate(self) -> float:
        """
        Calculate rate of progress (score improvement per step).
        
        Returns:
            Progress rate
        """
        if len(self.history) < 2:
            return 0.0
        
        first = self.history[0]
        last = self.history[-1]
        
        steps_delta = last.step - first.step
        if steps_delta == 0:
            return 0.0
        
        score_delta = last.score - first.score
        return score_delta / steps_delta
    
    def get_time_elapsed(self) -> float:
        """Get elapsed time since start."""
        if self.start_time is None:
            return 0.0
        
        end = self.end_time if self.end_time is not None else time.time()
        return end - self.start_time
    
    def get_summary(self) -> Dict[str, Any]:
        """Get progress summary."""
        if not self.history:
            return {"status": "no_progress"}
        
        current = self.history[-1]
        first = self.history[0]
        
        return {
            "total_steps": current.step,
            "current_score": current.score,
            "initial_score": first.score,
            "score_improvement": current.score - first.score,
            "progress_rate": self.get_progress_rate(),
            "time_elapsed": self.get_time_elapsed(),
            "criteria_met": current.criteria_met,
            "num_snapshots": len(self.history),
        }


class GoalMonitor:
    """
    Complete goal monitoring system for T-RLINKOS TRM++.
    
    Features:
    - Goal definition and tracking
    - Success criteria evaluation
    - Progress monitoring
    - Adaptive behavior based on progress
    """
    
    def __init__(self, goal: GoalDefinition):
        """
        Initialize goal monitor.
        
        Args:
            goal: Goal definition
        """
        self.goal = goal
        self.status = GoalStatus.NOT_STARTED
        self.tracker = ProgressTracker()
        
        self.criteria_history: Dict[str, List[bool]] = {
            c.name: [] for c in goal.success_criteria
        }
        
    def start(self):
        """Start monitoring goal."""
        self.status = GoalStatus.IN_PROGRESS
        self.tracker.start()
    
    def update(
        self,
        step: int,
        score: float,
        state: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update goal progress and check criteria.
        
        Args:
            step: Current step number
            score: Current score/performance
            state: Current state dictionary
            
        Returns:
            Update result with status and recommendations
        """
        if self.status != GoalStatus.IN_PROGRESS:
            return {"status": self.status.value, "message": "Goal not in progress"}
        
        # Check all criteria
        criteria_met = {}
        all_required_met = True
        weighted_score = 0.0
        total_weight = 0.0
        
        for criteria in self.goal.success_criteria:
            is_met = criteria.check(state)
            criteria_met[criteria.name] = is_met
            self.criteria_history[criteria.name].append(is_met)
            
            if is_met:
                weighted_score += criteria.weight
            total_weight += criteria.weight
            
            if criteria.is_required and not is_met:
                all_required_met = False
        
        # Calculate overall progress score
        progress_score = weighted_score / total_weight if total_weight > 0 else 0.0
        
        # Record progress
        self.tracker.record(
            step=step,
            score=score,
            criteria_met=criteria_met,
            metrics={"progress_score": progress_score}
        )
        
        # Check if goal achieved
        if all_required_met and progress_score >= 0.8:  # 80% of weighted criteria
            self.status = GoalStatus.ACHIEVED
            self.tracker.end()
            return {
                "status": self.status.value,
                "message": "Goal achieved!",
                "progress_score": progress_score,
                "criteria_met": criteria_met,
            }
        
        # Check for failure conditions
        failure_reasons = []
        
        if self.goal.max_steps and step >= self.goal.max_steps:
            failure_reasons.append(f"Exceeded max steps ({self.goal.max_steps})")
        
        if self.goal.max_time:
            elapsed = self.tracker.get_time_elapsed()
            if elapsed >= self.goal.max_time:
                failure_reasons.append(f"Exceeded max time ({self.goal.max_time}s)")
        
        if self.goal.min_score and score < self.goal.min_score:
            if step > 10:  # Allow some initial steps
                failure_reasons.append(f"Score below minimum ({self.goal.min_score})")
        
        if failure_reasons:
            self.status = GoalStatus.FAILED
            self.tracker.end()
            return {
                "status": self.status.value,
                "message": "Goal failed",
                "reasons": failure_reasons,
                "progress_score": progress_score,
            }
        
        # Generate recommendations based on progress
        recommendations = self._generate_recommendations(progress_score, criteria_met)
        
        return {
            "status": self.status.value,
            "progress_score": progress_score,
            "criteria_met": criteria_met,
            "recommendations": recommendations,
        }
    
    def _generate_recommendations(
        self,
        progress_score: float,
        criteria_met: Dict[str, bool]
    ) -> List[str]:
        """Generate recommendations based on progress."""
        recommendations = []
        
        # Check progress rate
        rate = self.tracker.get_progress_rate()
        if rate < 0:
            recommendations.append("Progress is declining - consider backtracking")
        elif rate < 0.01:
            recommendations.append("Slow progress - consider increasing exploration")
        
        # Check unmet required criteria
        unmet_required = [
            c.name for c in self.goal.success_criteria
            if c.is_required and not criteria_met.get(c.name, False)
        ]
        if unmet_required:
            recommendations.append(f"Focus on required criteria: {', '.join(unmet_required)}")
        
        # Check if stuck
        recent_history = self.tracker.history[-5:] if len(self.tracker.history) >= 5 else []
        if recent_history:
            scores = [s.score for s in recent_history]
            if max(scores) - min(scores) < 0.01:
                recommendations.append("Potentially stuck - consider different approach")
        
        return recommendations
    
    def get_status_report(self) -> Dict[str, Any]:
        """Get comprehensive status report."""
        report = {
            "goal_id": self.goal.goal_id,
            "description": self.goal.description,
            "status": self.status.value,
            "progress": self.tracker.get_summary(),
        }
        
        # Add criteria status
        criteria_status = []
        for criteria in self.goal.success_criteria:
            history = self.criteria_history.get(criteria.name, [])
            criteria_status.append({
                "name": criteria.name,
                "description": criteria.description,
                "is_required": criteria.is_required,
                "currently_met": history[-1] if history else False,
                "met_count": sum(history),
                "total_checks": len(history),
            })
        
        report["criteria"] = criteria_status
        
        return report
    
    def pause(self):
        """Pause goal monitoring."""
        if self.status == GoalStatus.IN_PROGRESS:
            self.status = GoalStatus.PAUSED
    
    def resume(self):
        """Resume goal monitoring."""
        if self.status == GoalStatus.PAUSED:
            self.status = GoalStatus.IN_PROGRESS


if __name__ == "__main__":
    # Test goal monitoring
    print("Testing Goal and Monitoring Pattern...")
    
    # Test 1: Define goal with criteria
    def score_threshold(state):
        return state.get("score", 0) > 0.8
    
    def steps_limit(state):
        return state.get("step", 0) < 100
    
    goal = GoalDefinition(
        goal_id="test_goal",
        description="Achieve high accuracy",
        success_criteria=[
            SuccessCriteria("high_score", "Score > 0.8", score_threshold, is_required=True),
            SuccessCriteria("efficient", "Steps < 100", steps_limit),
        ],
        max_steps=50,
    )
    
    monitor = GoalMonitor(goal)
    monitor.start()
    
    # Test 2: Simulate progress
    for i in range(20):
        score = 0.5 + i * 0.03  # Gradually improving
        state = {"score": score, "step": i}
        result = monitor.update(i, score, state)
        if i % 5 == 0:
            print(f"  Step {i}: status={result['status']}, progress={result.get('progress_score', 0):.2f}")
    
    # Test 3: Final status
    report = monitor.get_status_report()
    print(f"Test 3 - Final status: {report['status']}")
    print(f"         Progress: {report['progress']['score_improvement']:.2f}")
    
    # Test 4: Check criteria
    criteria_status = report['criteria']
    for c in criteria_status:
        print(f"  - {c['name']}: met={c['currently_met']} ({c['met_count']}/{c['total_checks']})")
    
    print("\n✅ Goal and Monitoring tests passed!")

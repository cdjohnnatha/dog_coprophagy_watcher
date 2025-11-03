"""
Finite State Machine for Ellie's behavior tracking.
Pure state transitions without I/O side effects.
"""
from dataclasses import dataclass
from typing import Optional, List
from enum import Enum
from .models import EllieState, Point


class Signal(str, Enum):
    """Signals that trigger state transitions."""
    DOG_IN_ZONE = "dog_in_zone"
    SQUAT_START = "squat_start"
    SQUAT_CONTINUE = "squat_continue"
    SQUAT_END = "squat_end"
    RESIDUE_CONFIRMED = "residue_confirmed"
    RESIDUE_NOT_FOUND = "residue_not_found"
    POOP_CLEANED = "poop_cleaned"
    COPROPHAGY_CONFIRMED = "coprophagy_confirmed"


@dataclass
class Command:
    """Command to be executed by adapters."""
    type: str
    data: dict


@dataclass
class StateTransition:
    """Result of a state transition."""
    new_state: EllieState
    commands: List[Command]
    
    def __init__(self, new_state: EllieState, commands: Optional[List[Command]] = None):
        self.new_state = new_state
        self.commands = commands or []


class EllieFSM:
    """Finite State Machine for Ellie's behavior."""
    
    def __init__(self, initial_state: EllieState = EllieState.IDLE):
        self.current_state = initial_state
    
    def transition(self, signal: Signal, context: dict) -> StateTransition:
        """
        Process a signal and return the new state with commands.
        
        Args:
            signal: The signal triggering the transition
            context: Additional context data for the transition
        
        Returns:
            StateTransition with new state and commands to execute
        """
        if signal == Signal.SQUAT_START:
            return self._handle_squat_start(context)
        
        elif signal == Signal.SQUAT_CONTINUE:
            return self._handle_squat_continue(context)
        
        elif signal == Signal.SQUAT_END:
            return self._handle_squat_end(context)
        
        elif signal == Signal.RESIDUE_CONFIRMED:
            return self._handle_residue_confirmed(context)
        
        elif signal == Signal.RESIDUE_NOT_FOUND:
            return self._handle_residue_not_found(context)
        
        elif signal == Signal.POOP_CLEANED:
            return self._handle_poop_cleaned(context)
        
        elif signal == Signal.COPROPHAGY_CONFIRMED:
            return self._handle_coprophagy_confirmed(context)
        
        # No transition
        return StateTransition(self.current_state, [])
    
    def _handle_squat_start(self, context: dict) -> StateTransition:
        """Handle squat start signal."""
        if self.current_state in (EllieState.IDLE, EllieState.DEFECACAO_CONFIRMADA):
            self.current_state = EllieState.POSSIVEL_DEFECACAO
            return StateTransition(
                EllieState.POSSIVEL_DEFECACAO,
                [
                    Command(
                        type="publish_state",
                        data={"state": EllieState.POSSIVEL_DEFECACAO.value}
                    ),
                    Command(
                        type="open_episode",
                        data={"timestamp": context.get("timestamp", 0.0)}
                    ),
                ]
            )
        return StateTransition(self.current_state, [])
    
    def _handle_squat_continue(self, context: dict) -> StateTransition:
        """Handle continued squatting."""
        if self.current_state == EllieState.POSSIVEL_DEFECACAO:
            duration = context.get("duration", 0.0)
            min_duration = context.get("min_duration", 5.0)
            
            if duration >= min_duration:
                self.current_state = EllieState.DEFECANDO
                return StateTransition(
                    EllieState.DEFECANDO,
                    [
                        Command(
                            type="publish_state",
                            data={"state": EllieState.DEFECANDO.value}
                        )
                    ]
                )
        return StateTransition(self.current_state, [])
    
    def _handle_squat_end(self, context: dict) -> StateTransition:
        """Handle end of squatting."""
        if self.current_state in (EllieState.POSSIVEL_DEFECACAO, EllieState.DEFECANDO):
            self.current_state = EllieState.AGUARDANDO_CONFIRMACAO
            return StateTransition(
                EllieState.AGUARDANDO_CONFIRMACAO,
                [
                    Command(
                        type="publish_state",
                        data={"state": EllieState.AGUARDANDO_CONFIRMACAO.value}
                    ),
                    Command(
                        type="start_confirmation_window",
                        data={"timestamp": context.get("timestamp", 0.0)}
                    ),
                ]
            )
        return StateTransition(self.current_state, [])
    
    def _handle_residue_confirmed(self, context: dict) -> StateTransition:
        """Handle confirmed residue detection."""
        if self.current_state == EllieState.AGUARDANDO_CONFIRMACAO:
            self.current_state = EllieState.DEFECACAO_CONFIRMADA
            
            commands = [
                Command(
                    type="publish_state",
                    data={"state": EllieState.DEFECACAO_CONFIRMADA.value}
                ),
                Command(
                    type="publish_poop_present",
                    data={
                        "value": True,
                        "zone": context.get("zone", ""),
                        "centroid": context.get("centroid", [0, 0]),
                        "area": context.get("area", 0),
                        "timestamp": context.get("timestamp", ""),
                    }
                ),
            ]
            
            # Add poop event if we have event_id
            if context.get("event_id"):
                commands.append(
                    Command(
                        type="publish_poop_event",
                        data={
                            "event_id": context["event_id"],
                            "camera": context.get("camera", ""),
                            "zone": context.get("zone", ""),
                            "timestamp": context.get("timestamp", ""),
                        }
                    )
                )
                commands.append(
                    Command(
                        type="update_frigate_sub_label",
                        data={
                            "event_id": context["event_id"],
                            "sub_label": context.get("sub_label", "poop"),
                        }
                    )
                )
            
            # Start monitoring
            commands.extend([
                Command(type="start_poop_monitor", data={}),
                Command(type="start_coprophagy_monitor", data={}),
            ])
            
            return StateTransition(self.current_state, commands)
        
        return StateTransition(self.current_state, [])
    
    def _handle_residue_not_found(self, context: dict) -> StateTransition:
        """Handle no residue found."""
        if self.current_state == EllieState.AGUARDANDO_CONFIRMACAO:
            self.current_state = EllieState.IDLE
            return StateTransition(
                EllieState.IDLE,
                [
                    Command(
                        type="publish_state",
                        data={"state": EllieState.IDLE.value}
                    )
                ]
            )
        return StateTransition(self.current_state, [])
    
    def _handle_poop_cleaned(self, context: dict) -> StateTransition:
        """Handle poop cleaned/removed."""
        if self.current_state == EllieState.DEFECACAO_CONFIRMADA:
            self.current_state = EllieState.IDLE
            
            commands = [
                Command(
                    type="publish_poop_present",
                    data={
                        "value": False,
                        "zone": context.get("zone", ""),
                        "timestamp": context.get("timestamp", ""),
                    }
                ),
                Command(
                    type="publish_state",
                    data={"state": EllieState.IDLE.value}
                ),
            ]
            
            # Check if coprophagy occurred during cleanup
            if context.get("dog_present", False):
                commands.append(
                    Command(
                        type="publish_coprophagy_event",
                        data={
                            "zone": context.get("zone", ""),
                            "timestamp": context.get("timestamp", ""),
                            "event_id": context.get("event_id"),
                        }
                    )
                )
                if context.get("event_id"):
                    commands.append(
                        Command(
                            type="update_frigate_sub_label",
                            data={
                                "event_id": context["event_id"],
                                "sub_label": "coprophagy",
                            }
                        )
                    )
            
            return StateTransition(self.current_state, commands)
        
        return StateTransition(self.current_state, [])
    
    def _handle_coprophagy_confirmed(self, context: dict) -> StateTransition:
        """Handle confirmed coprophagy."""
        if self.current_state == EllieState.DEFECACAO_CONFIRMADA:
            self.current_state = EllieState.COPROPHAGIA_CONFIRMADA
            
            commands = [
                Command(
                    type="publish_state",
                    data={"state": EllieState.COPROPHAGIA_CONFIRMADA.value}
                ),
                Command(
                    type="publish_coprophagy_event",
                    data={
                        "camera": context.get("camera", ""),
                        "zone": context.get("zone", ""),
                        "timestamp": context.get("timestamp", ""),
                        "event_id": context.get("event_id"),
                        "manual_event_id": context.get("manual_event_id"),
                    }
                ),
            ]
            
            return StateTransition(self.current_state, commands)
        
        return StateTransition(self.current_state, [])
    
    def get_state(self) -> EllieState:
        """Get current state."""
        return self.current_state
    
    def reset(self) -> None:
        """Reset to initial state."""
        self.current_state = EllieState.IDLE


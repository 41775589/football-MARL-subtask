"""Two of our players try to score from the edge of the box, one
is on the side with the ball, and unmarked. The other is at the
center, next to a defender, and facing the opponent keeper."""

from . import *

def build_scenario(builder):
  builder.config().game_duration = 400
  builder.config().deterministic = False
  builder.config().offsides = False
  builder.config().end_episode_on_score = True
  builder.config().end_episode_on_out_of_play = True
  builder.config().end_episode_on_possession_change = True
  builder.SetBallPosition(0.7, -0.28)

  builder.SetTeam(Team.e_Left)
  builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
  builder.AddPlayer(0.7, 0.0, e_PlayerRole_CB)
  builder.AddPlayer(0.7, -0.3, e_PlayerRole_CB)

  builder.SetTeam(Team.e_Right)
  builder.AddPlayer(-1.0, 0.0, e_PlayerRole_GK)
  builder.AddPlayer(-0.75, 0.1, e_PlayerRole_CB)

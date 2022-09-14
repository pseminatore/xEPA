import pandas as pd
import numpy
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px
import os
import plotly.colors as colors



def build_scores_as_of_week(scores, season, week, min, save_show=False):
    fig = go.Figure()
    players = scores['passer_player_name'].drop_duplicates().values.tolist()
    for player in players:
        data = scores.query('passer_player_name == @player')
        fig.add_trace(go.Scatter(x=data.week, y=data.xEPA, name=player, line=dict(width=8, color=data.team_color.values.tolist()[0])))
        fig.add_trace(go.Scatter(x=data.week, y=data.xEPA, name=player, line=dict(width=2, color=data.team_color2.values.tolist()[0])))
    if save_show:
        fig.show()
    else:
        if not os.path.exists(f"outputs/{season}"):
            os.mkdir(f"outputs/{season}")
        if not os.path.exists(f"outputs/{season}/{week}"):
            os.mkdir(f"outputs/{season}/{week}")
        fig.write_image(f"outputs/{season}/{week}/season_leaderboard.png")


def build_top_seasons_leaderboard(leaderboard, save_show=False):
    fig = go.Figure()
    leaderboard = leaderboard.reset_index(drop=True)
    leaderboard = leaderboard.reset_index()
    fig.add_trace(go.Table(
        header=dict(values=['Rank', 'Player', 'Season', 'xEPA'], height=56, font=dict(size=39)),
        cells=dict(values=[leaderboard.index + 1, leaderboard.passer_player_name, leaderboard.season, round(leaderboard.xEPA,1)], height=55, font=dict(size=24))
    ))
    title = f"Top xEPA Seasons"
    fig.add_annotation(xref="x domain",yref="paper",x=0.5, y=1.05, showarrow=False,
                text=title, font=dict(size=42))
    fig.add_annotation(xref="x domain",yref="paper",x=0.5, y=-0.025, showarrow=False,
                text='2009-2022 | Data from @nflfastr | Models,Graphic from @425k_football', font=dict(size=18))
    fig.update_layout(
        width=864,
        height=1600,
        template='ggplot2'
    ) 
    # for idx, logo in enumerate(leaderboard.headshot_url):
    #     fig.add_layout_image(
    #         x=0.375,
    #         xref='x domain',
    #         y=(0.97 - (idx * 0.0303)),
    #         yref='paper',
    #         sizex=0.1,
    #         sizey=0.1,
    #         layer='above',
    #         source=logo
    #     )
    if save_show:
        fig.show()
    else:
        fig.write_image(f"outputs/top_seasons_leaderboard.png")
        
def build_top_scores_by_season_leaderboard(leaderboard, season, week, min=10, save_show=False):
    fig = go.Figure()
    leaderboard = leaderboard.reset_index(drop=True)
    leaderboard = leaderboard.reset_index()
    fig.add_trace(go.Table(
        header=dict(values=['Rank', 'Player', 'xEPA'], height=56, font=dict(size=39)),
        cells=dict(values=[leaderboard.index + 1, leaderboard.passer_player_name, round(leaderboard.xEPA,1)], height=55, font=dict(size=24))
    ))
    title = f"xEPA Rankings Through Week {week}, {season}"
    footer = f"Min {min} Att. | Data from @nflfastr | Models,Graphic from @425k_football"
    fig.add_annotation(xref="x domain",yref="paper",x=0.5, y=1.05, showarrow=False,
                text=title, font=dict(size=42))
    fig.add_annotation(xref="x domain",yref="paper",x=0.5, y=-0.025, showarrow=False,
                text=footer, font=dict(size=18))
    fig.update_layout(
        width=1080,
        height=2000,
        template='ggplot2'
    ) 
    # for idx, logo in enumerate(leaderboard.headshot_url):
    #     fig.add_layout_image(
    #         x=0.375,
    #         xref='x domain',
    #         y=(0.97 - (idx * 0.0303)),
    #         yref='paper',
    #         sizex=0.1,
    #         sizey=0.1,
    #         layer='above',
    #         source=logo
    #     )
    if save_show:
        fig.show()
    else:
        if not os.path.exists(f"outputs/{season}"):
            os.mkdir(f"outputs/{season}")
        if not os.path.exists(f"outputs/{season}/{week}"):
            os.mkdir(f"outputs/{season}/{week}")
        fig.write_image(f"outputs/{season}/{week}/season_leaderboard.png")
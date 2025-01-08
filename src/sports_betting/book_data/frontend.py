from dash import Dash, dash_table, callback, html, dcc, Input, Output
import dash_bootstrap_components as dbc
import datetime as dt
import backend
import pandas as pd

external_stylesheets = [dbc.themes.BOOTSTRAP]
app = Dash(external_stylesheets=external_stylesheets)

# TODO: layout wise, I want to be able to toggle sports (mostly nba, nfl, cfb, cbb)
# ontop: config will be sport, date, stat
df = backend.build_prop_df(dt.date(2023, 5, 3))

def build_layout():
    layout = html.Div(
        [
            dbc.Row(
                [
                    html.H1('Prop Odds Visualization', style={'textAlign': 'center'})
                ]
            ),
            dbc.Row(
                [
                    dbc.Col(
                        dcc.Dropdown(
                            [
                                'nba',
                                'nfl',
                                'cfb',
                                'cbb'
                            ],
                            value='nba'
                        ),
                        style={'textAlign': 'center'}
                    ),
                    dbc.Col(
                        dcc.DatePickerSingle(
                            id='prop_date',
                            min_date_allowed=dt.date(2020, 1, 1),
                            max_date_allowed=dt.date(2024, 1, 1),
                            initial_visible_month=dt.date(2023, 5, 1),
                            date=dt.date(2023, 5, 3)
                        ),
                        style={'textAlign': 'center'},
                    ),
                    # TODO: update this with all possible markets
                    dbc.Col(
                        dcc.Dropdown(
                            [
                                'blocks',
                                'points'
                            ]
                        ),
                        style={'textAlign': 'center'}
                    )
                ]
            ),
            dbc.Row(
                [
                    dash_table.DataTable(
                            id='prop_table',
                            data=df.to_dict('records'),
                            page_size=15
                        )
                ]
            ),
            dbc.Row(
                [
                    html.Div(id='last_updated', style={'textAlign': 'center', 'marginTop': '20px'})
                ]
            ),
            dcc.Interval(
                id='interval_component',
                interval=60*1000,  # On the free API
                n_intervals=0
            )   
        ]
    )
    return layout
 
# TODO: make this pull the historical odds, saved in postgres
# per the api, starter key can be updated once every 30 minutes
@callback(
    [Output("prop_table", "data"),
    Output("last_updated", "children")],
    [Input("prop_date", "date"),
     Input("interval_component", "n_intervals")]
)
def update_table(date):
    if date is None:
        return []
    df = backend.build_prop_df(pd.to_datetime(date).date())
    last_updated = f"Last Updated: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    
    return df.to_dict('records'), last_updated

app.layout = build_layout()

# To run:
# python src/book_data/frontend.py
if __name__ == '__main__':
    app.run(debug=True)
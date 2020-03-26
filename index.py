import dash_core_components as dcc

import dash_html_components as html
from dash.dependencies import Input, Output

from app import app as app

from apps import index_page, app1, results, adjust115

app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])


@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/':
        return index_page.layout
    elif pathname == '/apps/app1':
        return app1.layout
    elif pathname == '/apps/results':
        return results.layout
    elif pathname == '/apps/adjust115':
        return adjust115.layout
    else:
        return '404'


if __name__ == '__main__':
    #app.run_server()
    app.run_server(debug=True, port=8080, host='0.0.0.0')

from dash import dcc
from dash import html

from dash.dependencies import Input, Output

from app import app

from apps import index_page, app1, results, adjust

app.layout = html.Div(
    [
        dcc.Location(id="url", refresh=False),
        html.Div(id="page-content", style={"height": "100vh"}),
    ]
)


@app.callback(Output("page-content", "children"), [Input("url", "pathname")])
def display_page(pathname):
    if pathname == "/":
        return index_page.layout
    elif pathname == "/apps/app1":
        return app1.layout
    elif pathname == "/apps/results":
        return results.layout
    elif pathname == "/apps/adjust":
        return adjust.layout
    else:
        return "404"


if __name__ == "__main__":
    app.run_server(debug=True, port=80, host="localhost")
    # app.run_server()
    # waitress-serve --port=80 --channel-timeout=400 --threads=6 index:app.server

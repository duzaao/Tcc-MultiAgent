#!/usr/bin/env python3
from agent import create_app

class Dummy:
    def run(self, m):
        return 'ok'

app = create_app(Dummy())
routes = [(r.path, sorted(list(r.methods))) for r in app.routes]
print('ROUTES:')
for p, m in routes:
    print(p, m)

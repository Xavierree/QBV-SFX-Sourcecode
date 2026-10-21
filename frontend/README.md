# Q-SFX frontend (Next.js)

The web interface of Q-SFX. It records a vocal imitation in the browser, sends the raw WAV bytes to the backend
(`../backend`, port 8500) and shows the ranked sound effects. It also hosts the Submit page, the Admin
dashboard and the OmniRoute metadata step. See the top-level `README.md` for the full setup.

```bash
npm install                  # from the repository root (installs the `frontend` workspace)
cp frontend/.env.local.example frontend/.env.local
cd frontend && npm run build && npm run start     # http://localhost:3000
```

Scripts: `npm run dev` (development), `npm run lint`, `npm run typecheck`, `npm test`.

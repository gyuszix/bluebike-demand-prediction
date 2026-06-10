# Combined image: builds the React frontend and serves it from the Node backend.
# The backend already serves ../frontend/build via express.static + a catch-all
# route, so one Cloud Run service hosts both the API and the UI (cheapest setup).
#
# Build context must be bluebikes-ui/ (so both frontend/ and backend/ are visible):
#   gcloud run deploy bluebikes-backend --source bluebikes-ui ...

# ---- Stage 1: build the React app ----
FROM node:18-alpine AS frontend
WORKDIR /fe
COPY frontend/package*.json ./
RUN npm install --no-audit --no-fund
COPY frontend/ ./
# Serve from root (override package.json "homepage"), talk to same-origin /api,
# point the monitoring dashboard at the prediction service. CI=false so lint
# warnings don't fail the production build.
ENV PUBLIC_URL=/ \
    CI=false \
    REACT_APP_API_URL=/api \
    REACT_APP_MONITORING_URL=https://bluebikes-prediction-784495516002.us-central1.run.app
RUN npm run build

# ---- Stage 2: backend that serves the built frontend ----
FROM node:18-alpine
WORKDIR /app
COPY backend/package*.json ./
RUN npm install --production
COPY backend/ ./
# server.js resolves the UI at path.join(__dirname, '../frontend/build') => /frontend/build
COPY --from=frontend /fe/build /frontend/build
ENV PORT=8080
EXPOSE 8080
CMD ["npm", "run", "start"]

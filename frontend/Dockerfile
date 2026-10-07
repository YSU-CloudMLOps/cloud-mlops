# ==========================================
# Stage 1: Build the React + Vite frontend
# ==========================================
FROM node:20-alpine AS builder

WORKDIR /app

# Install dependencies (cached when package files don't change)
COPY package.json package-lock.json* ./
RUN npm ci || npm install

# Copy application source code and compile
COPY . .
RUN npm run build

# ==========================================
# Stage 2: Production Nginx web server
# ==========================================
FROM nginx:alpine

# Remove default Nginx welcome site
RUN rm -rf /usr/share/nginx/html/*

# Copy compiled static assets from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy custom Nginx reverse proxy configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Expose standard HTTP port
EXPOSE 80

# Health check to verify Nginx is actively responding
HEALTHCHECK --interval=20s --timeout=5s --retries=3 \
  CMD wget -qO- http://localhost/ || exit 1

CMD ["nginx", "-g", "daemon off;"]

FROM node:24-alpine

WORKDIR /app/frontend

# Install pnpm
RUN corepack enable && corepack prepare pnpm@latest --activate

COPY package.json .

RUN pnpm install

COPY . .
FROM node:24-alpine

WORKDIR /app/frontend

# Install pnpm
RUN corepack enable && corepack prepare pnpm@latest --activate

COPY package.json  .
COPY pnpm-lock.yaml .

RUN pnpm install

COPY . .

EXPOSE 3000
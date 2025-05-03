# Tensara 

A platform for GPU programming challenges. Write efficient GPU kernels and compare your solutions with other developers!

## Getting Started

### Prerequisites

- Node.js 18+ and npm/yarn/pnpm
- PostgreSQL (for local development) or a Supabase account
- Git

### Environment Setup

1. Clone the repository:
```bash
git clone https://github.com/tensara/tensara
cd tensara
```

2. Install dependencies:
```bash
npm install
# or
yarn
# or
pnpm install
```

3. Set up your environment variables by copying the example file:
```bash
cp .env.example .env
```

### Database Setup

You have a few options for setting up the database:

#### Option 1: Local PostgreSQL

1. Install PostgreSQL on your machine if you haven't already:
   - macOS (using Homebrew): `brew install postgresql`
   - Linux: `sudo apt-get install postgresql`
   - Windows: Download from [PostgreSQL website](https://www.postgresql.org/download/windows/)

2. Start PostgreSQL service:
   - macOS: `brew services start postgresql`
   - Linux: `sudo service postgresql start`
   - Windows: It should start automatically as a service

3. Create a new database:
```bash
createdb tensara_db
```

4. Update your `.env` file with local PostgreSQL connection string:
```
DATABASE_URL="postgresql://your_username@localhost:5432/tensara_db"
```

#### Option 2: Supabase (Hosted PostgreSQL)

1. Create a new project on [Supabase](https://supabase.com)

2. Once your project is created, go to Settings > Database to find your connection string

3. Update your `.env` file with the Supabase connection string:
```
DATABASE_URL="postgresql://postgres:[YOUR-PASSWORD]@db.[YOUR-PROJECT-REF].supabase.co:5432/postgres"
```

### Final Setup Steps

1. Push the database schema:
```bash
npx prisma db push
```

2. Generate Prisma Client:
```bash
npx prisma generate
```
3. Push the problems to the database. Follow the steps in the [problems](https://github.com/tensara/problems/) repository. 
4. Start the development server:
```bash
npm run dev
# or
yarn dev
# or
pnpm dev
```

Your app should now be running at [http://localhost:3000](http://localhost:3000)!

## Environment Variables

Create a `.env` file in the root directory with the following variables:

```env
# Database connection string
DATABASE_URL="your_connection_string_here"

# NextAuth configuration
NEXTAUTH_SECRET="your_nextauth_secret"
AUTH_GITHUB_ID=""
AUTH_GITHUB_SECRET=""
NEXTAUTH_URL="http://localhost:3000"

# Google Analytics ID (can be ignored until production)
NEXT_PUBLIC_GA_ID=""

MODAL_CHECKER_SLUG=""
MODAL_BENCHMARK_SLUG=""

MODAL_ENDPOINT=""
```

## Sponsors

Thank you to our sponsors who help make tensara possible:

- [Modal](https://www.modal.com?utm_source=github&utm_medium=github&utm_campaign=tensara) - Modal lets you run
jobs in the cloud, by just writing a few lines of Python. Customers use Modal to deploy Gen AI models at large scale,
fine-tune large language models, run protein folding simulations, and much more.

We use Modal to securely run accurate benchmarks on various GPUs.

Interested in sponsoring? Contact us at [sponsor@tensara.org](mailto:sponsor@tensara.org)


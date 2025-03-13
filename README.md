# Tensara App

Tensara is a competitive programming platform specifically designed for CUDA kernel optimization challenges, similar to Codeforces but focused on GPU computing. It provides developers with a space to compete, learn, and improve their CUDA programming skills through real-world optimization problems.

### Key Features
- 🚀 Compete in CUDA kernel optimization challenges
- 💻 Practice problems like Matrix Multiplication, Leaky ReLU, and other GPU computing tasks
- 📊 Real-time performance benchmarking and scoring
- 📈 Leaderboards to track your ranking against other developers
- 🔍 View and learn from other developers' optimized solutions
- ⚡ Automated testing and performance measurement
- 📱 Browse problems and standings from any device

This is a modern web application built with the [T3 Stack](https://create.t3.gg/), combining the power of Next.js, tRPC, Prisma, and more.

## 🚀 Getting Started

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

3. Start the development server:
```bash
npm run dev
# or
yarn dev
# or
pnpm dev
```

Your app should now be running at [http://localhost:3000](http://localhost:3000)!

## 📝 Environment Variables

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

## 🛠 Tech Stack

- [Next.js](https://nextjs.org) - React framework
- [NextAuth.js](https://next-auth.js.org) - Authentication
- [Prisma](https://prisma.io) - Database ORM
- [Chakra UI](https://v2.chakra-ui.com/) - Styling
- [tRPC](https://trpc.io) - End-to-end typesafe APIs

## 📚 Learn More

To learn more about the [T3 Stack](https://create.t3.gg/), check out:

- [T3 Documentation](https://create.t3.gg/)
- [Learn the T3 Stack](https://create.t3.gg/en/faq#what-learning-resources-are-currently-available)

## 🚀 Deployment

For deployment instructions, follow our guides for:
- [Vercel](https://create.t3.gg/en/deployment/vercel)
- [Netlify](https://create.t3.gg/en/deployment/netlify)
- [Docker](https://create.t3.gg/en/deployment/docker)

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

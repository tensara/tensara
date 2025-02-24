import NextAuth from "next-auth";
import { authConfig } from "./config";

// eslint-disable-next-line @typescript-eslint/no-unsafe-assignment
export const handlers = NextAuth(authConfig);

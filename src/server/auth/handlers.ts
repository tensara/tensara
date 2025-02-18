import { NextAuthOptions } from "next-auth";
import NextAuth from "next-auth";
import { authConfig } from "./config";

export const handlers = NextAuth(authConfig as NextAuthOptions);

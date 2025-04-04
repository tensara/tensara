import { type Session } from "next-auth";
import { SessionProvider } from "next-auth/react";
import { type AppType } from "next/app";
import { Providers } from "~/components/layout/providers";
import { api } from "~/utils/api";

import "~/styles/globals.css";
import "katex/dist/katex.min.css";
import "highlight.js/styles/github-dark.css";

const MyApp: AppType<{ session: Session | null }> = ({
  Component,
  pageProps: { session, ...pageProps },
}) => {
  return (
    <SessionProvider session={session}>
      <Providers>
        <Component {...pageProps} />
      </Providers>
    </SessionProvider>
  );
};

export default api.withTRPC(MyApp);

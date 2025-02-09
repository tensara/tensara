import { GeistSans } from "geist/font/sans";
import { type Session } from "next-auth";
import { SessionProvider } from "next-auth/react";
import { type AppType } from "next/app";
import { Providers } from "~/components/layout/providers";

import { api } from "~/utils/api";

import "~/styles/globals.css";

const MyApp: AppType<{ session: Session | null }> = ({
  Component,
  pageProps: { session, ...pageProps },
}) => {
  return (
    <SessionProvider session={session}>
      <Providers>
        <div className={GeistSans.className}>
          <Component {...pageProps} />
        </div>
      </Providers>
    </SessionProvider>
  );
};

export default api.withTRPC(MyApp);

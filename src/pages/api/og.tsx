import { ImageResponse } from "@vercel/og";
import { type NextRequest } from "next/server";

export const config = {
  runtime: "experimental-edge",
};

export default async function handler(req: NextRequest) {
  const { searchParams } = new URL(req.url);

  const title = searchParams.get("title") ?? "Tensara";
  const subTitle = searchParams.get("subTitle") ?? "";

  // Load the logo and font
  const [logoData, DMSansData, SpaceGroteskData] = await Promise.all([
    fetch(new URL("../../../public/logo_og.png", import.meta.url)).then((res) =>
      res.arrayBuffer()
    ),
    fetch(
      new URL("../../../public/DMSans_24pt-SemiBold.ttf", import.meta.url)
    ).then((res) => res.arrayBuffer()),
    fetch(
      new URL("../../../public/SpaceGrotesk-Medium.ttf", import.meta.url)
    ).then((res) => res.arrayBuffer()),
  ]);

  return new ImageResponse(
    (
      <div
        style={{
          height: "100%",
          width: "100%",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          backgroundColor: "#1E293B", // gray.900
          padding: "40px",
        }}
      >
        {/* Logo */}
        <img
          src={`data:image/png;base64,${Buffer.from(logoData).toString("base64")}`}
          alt="Tensara Logo"
          width={200}
          height={200}
          style={{ objectFit: "contain", marginBottom: "20px" }}
        />

        {/* Title */}
        <div
          style={{
            fontSize: "65px",
            fontFamily: "DM Sans",
            fontWeight: 600,
            color: "white",
            marginBottom: "15px",
            textAlign: "center",
            maxWidth: "1000px",
          }}
        >
          {title}
        </div>

        {/* Subtitle */}
        {subTitle && (
          <div
            style={{
              fontFamily: "Space Grotesk",
              fontSize: "25px",
              color: "white",
              textAlign: "center",
              maxWidth: "800px",
            }}
          >
            {subTitle}
          </div>
        )}
      </div>
    ),
    {
      width: 1200,
      height: 630,
      fonts: [
        {
          name: "DM Sans",
          data: DMSansData,
          weight: 600,
          style: "normal",
        },
        {
          name: "Space Grotesk",
          data: SpaceGroteskData,
          weight: 300,
          style: "normal",
        },
      ],
    }
  );
  // } catch (e) {
  //   console.error(e);
  //   return new Response(`Failed to generate image: ${e instanceof Error ? e.message : 'Unknown error'}`, { status: 500 });
  // }
}

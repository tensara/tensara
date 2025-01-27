import { createSystem, defaultConfig, defineRecipe } from "@chakra-ui/react";

export const system = createSystem(defaultConfig, {
  theme: {
    tokens: {
      fonts: {
        body: { value: "Gotham, sans-serif" }, // Fixed font name and syntax
      },
      colors: {
        brand: {
          50: { value: "#011518" },
          100: { value: "#022227" },
          400: { value: "#04424B" },
          600: { value: "#4FD1C5" },
          700: { value: "#DAB785" },
        },
      },
    },
    semanticTokens: {
      colors: {
        "chakra-body-bg": { value: "{colors.brand.50}" }, // Fixed reference syntax
        "chakra-body-text": { value: "white" },
      },
    },
    recipes: {
      button: defineRecipe({
        base: {
          borderRadius: "md",
          fontWeight: "medium",
        },
      }),
      input: defineRecipe({
        base: {
          bg: "brand.100",
          color: "white",
          _placeholder: {
            color: "brand.400", // Fixed placeholder syntax
          },
        },
      }),
    },
  },
});

import { extendTheme } from "@chakra-ui/react";

// Here are 5 excellent font options - uncomment the one you want to try
export const system = extendTheme({
  fonts: {
    // Inter
    // body: "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Oxygen, Ubuntu, Cantarell, Fira Sans, Droid Sans, Helvetica Neue, sans-serif",
    // heading: "Inter, -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Oxygen, Ubuntu, Cantarell, Fira Sans, Droid Sans, Helvetica Neue, sans-serif",

    // DM Sans
    body: "DM Sans, -apple-system, BlinkMacSystemFont, system-ui, sans-serif",
    heading:
      "DM Sans, -apple-system, BlinkMacSystemFont, system-ui, sans-serif",

    // Onest
    // body: "Onest, -apple-system, BlinkMacSystemFont, system-ui, sans-serif",
    // heading: "Onest, -apple-system, BlinkMacSystemFont, system-ui, sans-serif",
  },
  styles: {
    global: {
      body: {
        bg: "#0F172A",
        color: "white",
      },
    },
  },
  colors: {
    brand: {
      primary: "#10B981",
      secondary: "#181f2a",
      dark: "#101723",
      navbar: "#0e8144",
      card: "rgba(15, 23, 42, 0.6)",
    },
    gray: {
      700: "#334155",
      800: "#1E293B",
      900: "#0F172A",
    },
  },
  components: {
    Button: {
      baseStyle: {
        fontWeight: "medium",
        borderRadius: "lg",
      },
      variants: {
        solid: {
          bg: "brand.primary",
          color: "white",
          _hover: {
            bg: "brand.primary",
            opacity: 0.9,
          },
        },
        ghost: {
          color: "whiteAlpha.900",
          _hover: {
            bg: "whiteAlpha.200",
          },
        },
      },
    },
    Input: {
      variants: {
        filled: {
          field: {
            bg: "whiteAlpha.100",
            borderRadius: "lg",
            _hover: {
              bg: "whiteAlpha.200",
            },
            _focus: {
              bg: "whiteAlpha.200",
              borderColor: "brand.primary",
            },
          },
        },
      },
      defaultProps: {
        variant: "filled",
      },
    },
    Container: {
      baseStyle: {
        maxW: "8xl",
      },
    },
  },
});

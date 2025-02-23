import { extendTheme } from "@chakra-ui/react";

export const system = extendTheme({
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
      secondary: "#1E293B",
      dark: "#0F172A",
      navbar: "#065F46",
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

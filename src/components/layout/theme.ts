import { extendTheme } from "@chakra-ui/react";

export const system = extendTheme({
  styles: {
    global: {
      body: {
        bg: "#0A1628",
        color: "white",
      },
    },
  },
  colors: {
    brand: {
      primary: "#1DB954",
      secondary: "#1A2C42",
      dark: "#0A1628",
      sidebar: "#162A46",
    },
    gray: {
      700: "#2D3748",
      800: "#1A2C42",
      900: "#0A1628",
    },
  },
  components: {
    Button: {
      baseStyle: {
        fontWeight: "medium",
        borderRadius: "full",
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
            bg: "whiteAlpha.200",
            borderRadius: "full",
            _hover: {
              bg: "whiteAlpha.300",
            },
            _focus: {
              bg: "whiteAlpha.300",
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

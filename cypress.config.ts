import { defineConfig } from "cypress";

export default defineConfig({
  projectId: "3p1ow3",
  e2e: {
    setupNodeEvents(on, config) {
      // implement node event listeners here
    },
    chromeWebSecurity: false,
    baseUrl: 'http://localhost:3000',
    video: false
  },
});

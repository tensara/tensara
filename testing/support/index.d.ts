/// <reference types="cypress" />

declare namespace Cypress {
  interface Chainable<> {
    setEditorContent(content: string): Chainable<void>
    login(): Chainable<void>
    loginWithGithub(username: string, password: string): Chainable<void>
  }
}


declare global {
  interface Window {
    monaco: {
      editor: {
        getModels: () => Array<{
          setValue: (content: string) => void;
        }>;
      };
    };
  }
}

Cypress.Commands.add('setEditorContent', (content: string) => {
  cy.window().then(win => {
    if (win.monaco?.editor) {
      const editor = win.monaco.editor.getModels()[0];
      if (editor) {
        editor.setValue(content);
      }
    }
  });
});

Cypress.Commands.add('loginWithGithub', (username: string, password: string) => {
  cy.visit('/');
  
  cy.visit('/api/auth/signin/github');
  cy.get('button[type="submit"]').click({ force: true });
  cy.origin("https://github.com", { args: { username, password } }, ({ username, password }) => {
    cy.get('input[name="login"]').should('be.visible').type(username, { force: true});
    cy.get('input[name="password"]').should('be.visible').type(password, { force: true });
    cy.get('input[type="submit"]').click({ force: true });
    cy.wait(1000);
  });

    
  cy.get('body').then(($body) => {
    if ($body.find('button:contains("Authorize")').length > 0) {
      cy.get('button[type="submit"]').contains('Authorize').click();
    }
  });



  
});

export {};
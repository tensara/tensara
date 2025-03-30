describe('Tensara Homepage', () => {
  beforeEach(() => {
    cy.visit('/');
  });

  it('should display the main description text', () => {
    cy.contains('A platform for GPU programming challenges').should('be.visible');
  });

  it('should have a working "Start solving" button', () => {
    cy.get('a[href="/problems"]').contains('Start solving').should('be.visible').click();
    cy.url().should('include', '/problems');
  });
});

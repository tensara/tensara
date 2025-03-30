// cypress/e2e/problem-navigation.cy.js

describe('Problem Navigation Tests', () => {
  beforeEach(() => {
    // Start each test from the homepage
    
  });

  it('Navigate to problems page from homepage', () => {
    // Click the Start solving button on the homepage
    cy.visit('/');

    cy.get('a[href="/problems"]').contains('Start solving').click();
    
    // Verify we've reached the problems page
    cy.url().should('include', '/problems');
    
    // Verify the problems page has text "Vector Addition", "Matrix Multiplication" 
    cy.contains('Vector Addition').should('exist');
    cy.contains('Matrix Multiplication').should('exist');
    cy.contains('ReLU').should('exist');
  });

  it('should find and click on the ReLU problem', () => {
    // Navigate to problems page
    cy.visit('/');
    
    // Click on the "Start solving" button to go to problems page
    cy.get('a[href="/problems"]').contains('Start solving').click();
    
    // Verify we're on the problems page
    cy.url().should('include', '/problems');
    
    // Find and click on the ReLU problem (exact match)
    cy.contains(/^ReLU$/).click();
    
    // Verify we've navigated to the ReLU problem page
    cy.url().should('include', '/problems/');
    
    // Verify the ReLU problem title is visible
    cy.contains('ReLU').should('be.visible');
  });

  it('should display proper content on the ReLU problem page', () => {
    // Navigate to problems page
    cy.visit('/problems/relu');
    
    // Check for key elements on the problem page
    cy.contains('ReLU').should('be.visible');
    
    // Code editor should be visible
    cy.get('.monaco-editor').should('exist').and('be.visible');
    
    // 2. Check for editor content area
    cy.get('.monaco-editor-background').should('exist');
    

    // Submit button should be visible
    cy.contains('button', 'Submit', { matchCase: false }).should('exist');
  });

});
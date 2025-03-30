// cypress/e2e/problem-navigation.cy.js

import { ReluSolutions } from "cypress/support/solutions";

describe('Problem Navigation Tests', () => {
  beforeEach(() => {
    // Start each test from the homepage
    cy.visit('/');
    cy.loginWithGithub(Cypress.env('USERNAME') as string, Cypress.env('PASSWORD') as string);
    cy.wait(1000);
    
  });

  it('submits correct solution', () => {

    cy.visit('/problems/relu');
    // Get the correct solution from the fixtures
    const correct_solution = ReluSolutions.correct;
    //wait for editor to load
    cy.get('.monaco-editor').should('exist');
    //wait for editor to be visible
    cy.get('.monaco-editor').should('be.visible');

    cy.setEditorContent(correct_solution);

    //get button with text "Submit"
    cy.contains('button', 'Submit', { matchCase: false }).should('exist');

    //click submit button
    cy.contains('button', 'Submit', { matchCase: false }).click();

    // wait for 30 seconds or until ACCEPTED text appears
    cy.contains('ACCEPTED', { timeout: 30000 }).should('exist');
  });

  it('submits compile error solution', () => {
    cy.visit('/problems/relu');
    // Get the compile error solution from the fixtures
    const compile_error_solution = ReluSolutions.compile_error;
    //wait for editor to load
    cy.get('.monaco-editor').should('exist');
    //wait for editor to be visible
    cy.get('.monaco-editor').should('be.visible');

    cy.setEditorContent(compile_error_solution);

    //get button with text "Submit"
    cy.contains('button', 'Submit', { matchCase: false }).should('exist');

    //click submit button
    cy.contains('button', 'Submit', { matchCase: false }).click();

    // wait for 30 seconds or until COMPILATION ERROR text appears
    cy.contains('Compile Error', { timeout: 30000 }).should('exist');
  });

  it('submits runtime error solution', () => {
    cy.visit('/problems/relu');
    // Get the runtime error solution from the fixtures
    const runtime_error_solution = ReluSolutions.runtime_error;
    //wait for editor to load
    cy.get('.monaco-editor').should('exist');
    //wait for editor to be visible
    cy.get('.monaco-editor').should('be.visible');

    cy.setEditorContent(runtime_error_solution);

    //get button with text "Submit"
    cy.contains('button', 'Submit', { matchCase: false }).should('exist');

    //click submit button
    cy.contains('button', 'Submit', { matchCase: false }).click();

    // wait for 30 seconds or until RUNTIME ERROR text appears
    cy.contains('Error', { timeout: 30000 }).should('exist');
  });

  it('submits wrong answer solution', () => {
    cy.visit('/problems/relu');
    // Get the runtime error solution from the fixtures
    const wrong_answer_solution = ReluSolutions.wrong_answer;
    //wait for editor to load
    cy.get('.monaco-editor').should('exist');
    //wait for editor to be visible
    cy.get('.monaco-editor').should('be.visible');

    cy.setEditorContent(wrong_answer_solution);

    //get button with text "Submit"
    cy.contains('button', 'Submit', { matchCase: false }).should('exist');

    //click submit button
    cy.contains('button', 'Submit', { matchCase: false }).click();

    // wait for 30 seconds or until WRONG ANSWER text appears
    cy.contains('Wrong Answer', { timeout: 30000 }).should('exist');
    
  });
});
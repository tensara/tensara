import { Conv1DSolutions } from "testing/support/solutions";

describe("Conv1D Problem Tests", () => {
  beforeEach(() => {
    // Start each test from the homepage
    cy.visit("/");
    cy.loginWithGithub(
      Cypress.env("USERNAME") as string,
      Cypress.env("PASSWORD") as string
    );
    cy.wait(1000);
  });

  it("submits correct solution", () => {
    cy.visit("/problems/conv-1d");
    // Get the correct solution from the fixtures
    const correct_solution = Conv1DSolutions.correct;
    //wait for editor to load
    cy.get(".monaco-editor").should("exist");
    //wait for editor to be visible
    cy.get(".monaco-editor").should("be.visible");

    cy.setEditorContent(correct_solution);

    //get button with text "Submit"
    cy.contains("button", "Submit", { matchCase: false }).should("exist");

    //click submit button
    cy.contains("button", "Submit", { matchCase: false }).click();

    // wait for 30 seconds or until ACCEPTED text appears
    cy.contains("ACCEPTED", { timeout: 30000 }).should("exist");
  });

  it("submits compile error solution", () => {
    if (Conv1DSolutions.compile_error.trim() === "") {
      return;
    }
    cy.visit("/problems/conv-1d");
    // Get the compile error solution from the fixtures
    const compile_error_solution = Conv1DSolutions.compile_error;
    //wait for editor to load
    cy.get(".monaco-editor").should("exist");
    //wait for editor to be visible
    cy.get(".monaco-editor").should("be.visible");

    cy.setEditorContent(compile_error_solution);

    //get button with text "Submit"
    cy.contains("button", "Submit", { matchCase: false }).should("exist");

    //click submit button
    cy.contains("button", "Submit", { matchCase: false }).click();

    // wait for 30 seconds or until COMPILATION ERROR text appears
    cy.contains("Compile Error", { timeout: 30000 }).should("exist");
  });

  it("submits runtime error solution", () => {
    if (Conv1DSolutions.runtime_error.trim() === "") {
      it.skip("No runtime error solution provided");
    }
    cy.visit("/problems/conv-1d");
    // Get the runtime error solution from the fixtures
    const runtime_error_solution = Conv1DSolutions.runtime_error;
    //wait for editor to load
    cy.get(".monaco-editor").should("exist");
    //wait for editor to be visible
    cy.get(".monaco-editor").should("be.visible");

    cy.setEditorContent(runtime_error_solution);

    //get button with text "Submit"
    cy.contains("button", "Submit", { matchCase: false }).should("exist");

    //click submit button
    cy.contains("button", "Submit", { matchCase: false }).click();

    // wait for 30 seconds or until RUNTIME ERROR text appears
    cy.contains("Error", { timeout: 30000 }).should("exist");
  });

  it("submits wrong answer solution", () => {
    if (Conv1DSolutions.wrong_answer.trim() === "") {
      it.skip("No wrong answer solution provided");
    }
    cy.visit("/problems/conv-1d");
    // Get the runtime error solution from the fixtures
    const wrong_answer_solution = Conv1DSolutions.wrong_answer;
    //wait for editor to load
    cy.get(".monaco-editor").should("exist");
    //wait for editor to be visible
    cy.get(".monaco-editor").should("be.visible");

    cy.setEditorContent(wrong_answer_solution);

    //get button with text "Submit"
    cy.contains("button", "Submit", { matchCase: false }).should("exist");

    //click submit button
    cy.contains("button", "Submit", { matchCase: false }).click();

    // wait for 30 seconds or until WRONG ANSWER text appears
    cy.contains("Wrong Answer", { timeout: 30000 }).should("exist");
  });
});

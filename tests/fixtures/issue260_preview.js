// preview.js - module for previewing saved survey data

import SurveyPreview from './common/SurveyPreview.js';
import Question from './common/Question.js';
import SingleSelectionQuestion from './common/SingleSelectionQuestion.js';
import MultiSelectionQuestion from './common/MultiSelectionQuestion.js';
import RatingQuestion from './common/RatingQuestion.js';

document.addEventListener('DOMContentLoaded', () => {
  // Find the form on the preview page
  const form = document.querySelector('form');
  if (!form) {
    console.error('Preview: No form element found');
    return;
  }

  // Load saved survey data from localStorage
  let data = null;
  try {
    if (localStorage.data) {
      data = JSON.parse(localStorage.data);
    }
  } catch (e) {
    console.error('Failed to parse survey data from localStorage:', e);
  }

  // Create SurveyPreview instance with title from stored data (if any)
  const previewTitle = (data && typeof data.title === 'string') ? data.title : '';
  const preview = new SurveyPreview('form', previewTitle); // title may be used for future extensions

  if (data && Array.isArray(data.questions)) {
    // For each question, create a Question instance in preview mode (preview=true)
    data.questions.forEach(q => {
      const title = q.title || 'Untitled';
      const name = q.name || `q-${Date.now()}-${Math.random().toString(36).substr(2,5)}`;
      let question;
      if (q.className === 'SingleSelectionQuestion') {
        const opts = Array.isArray(q.options) ? q.options : [];
        question = new SingleSelectionQuestion(title, name, true, opts);
      } else if (q.className === 'MultiSelectionQuestion') {
        const opts = Array.isArray(q.options) ? q.options : [];
        question = new MultiSelectionQuestion(title, name, true, opts);
      } else if (q.className === 'OpenQuestion') {
        const isMultilines = typeof q.isMultilines === 'boolean' ? q.isMultilines : false;
        question = new OpenQuestion(title, name, true, isMultilines);
      } else if (q.className === 'RatingQuestion') {
        const starCount = typeof q.starCount === 'number' ? q.starCount : undefined;
        question = new RatingQuestion(title, name, true, starCount);
      } else {
        question = new Question(title, name, true);
      }
      preview.appendQuestion(question);
    });
  } else {
    console.warn('No survey data found in localStorage or invalid format');
  }

  // Submit handler (optional, just logs)
  form.addEventListener('submit', (e) => {
    console.log('Preview form submitted');
    // default navigation to "submit" action proceeds
  });

  console.log('Preview page initialized');
});
